"""
LoRA fine-tuning of Gemma 3 270M for legal document metadata extraction.

Install dependencies first:
    pip install -U "transformers>=4.40" datasets accelerate peft trl bitsandbytes tensorboard matplotlib

Training data layout
────────────────────
DOC_DIR      — folder containing source documents (.pdf / .docx / .doc / .txt)
LABELS_JSON  — JSON file: list of ground-truth records, each with:
    {
      "file_path":      "/abs/or/relative/path/to/doc.pdf",
      "title":          "MOTION TO DISMISS",          // null if unknown
      "document_types": ["Motion to Dismiss"],
      "nuid":           "1:23-CV-04521-RJS",           // null if unknown
      "court_name":     "UNITED STATES DISTRICT COURT",
      "court_location": "Southern District of New York",
      "judge_name":     "Robert J. Sullivan",          // null if unknown
      "filing_date":    "2023-09-28",                  // null if unknown
      "parties": {
        "plaintiffs":  ["Alice Corp."],
        "defendants":  ["Bob Inc."],
        "petitioners": [],
        "respondents": []
      },
      "clauses": ["Fourth Amendment", "Due Process Clause"]
    }

Usage:
    # Train
    python scripts/train_lora.py

    # Inference with trained adapter — single document
    python scripts/train_lora.py --infer /path/to/doc.pdf

    # Inference on a whole folder (results → gemma3-270m-lora-adapter/inference_results.json)
    python scripts/train_lora.py --infer-folder /path/to/docs/

    # Inspect which layers have adapters
    python scripts/train_lora.py --inspect-layers

    # Plot training curves from a finished run
    python scripts/train_lora.py --visualize
    python scripts/train_lora.py --visualize --viz-out plots/

    # Launch TensorBoard interactively
    tensorboard --logdir gemma3-270m-lora-adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths — fill these in
# ---------------------------------------------------------------------------

DOC_DIR    = Path("/home/ishaan/work/docFilter/EPONAMix-Inference/Predicted_Relevant")          # folder with source docs
LABELS_JSON = Path("data/groq_labels/ground_truth.json") # ground-truth annotation file
OUT_DIR    = Path("./gemma3-270m-lora-adapter")

BASE_MODEL = "google/gemma-3-270m-it"   # instruction-tuned base; better for SFT
FRONT_PAGES    = 2       # how many pages from the start to include
BACK_PAGES     = 2       # how many pages from the end to include
MAX_SEQ_LEN    = 512     # tokens per sequence — keep low for 4 GB GPU

# ---------------------------------------------------------------------------
# Prompt templates  (identical to llm_extract.py so inference is consistent)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a legal document metadata extractor.
Read the document excerpt and fill in the JSON template exactly.
Output ONLY the completed JSON — no explanation, no prose, no markdown fences.\
"""

_JSON_SKELETON = """\
{
  "title": null,
  "document_types": [],
  "nuid": null,
  "court_name": null,
  "court_location": null,
  "judge_name": null,
  "filing_date": null,
  "parties": {"plaintiffs": [], "defendants": [], "petitioners": [], "respondents": []},
  "clauses": []
}\
"""

_FIELD_HINTS = """\
Field definitions:
- title: the document's own title in ALL CAPS (e.g. "MOTION TO DISMISS"), or null
- document_types: list of types, e.g. ["Motion to Dismiss", "Memorandum"]
- nuid: normalised docket/case number, e.g. "1:23-CV-04521-RJS", or null
- court_name: full court name, e.g. "UNITED STATES DISTRICT COURT", or null
- court_location: district or county, e.g. "Southern District of New York", or null
- judge_name: presiding judge without "Hon.", e.g. "Robert J. Sullivan", or null
- filing_date: date as YYYY-MM-DD from the signature block "Dated:" line, or null
- parties.plaintiffs / defendants: names from the caption above/below "v.", or []
- clauses: US Constitutional provisions cited, e.g. ["Fourth Amendment"], or []\
"""


def _extract_page_window(zones) -> str:
    """
    Return text from the first FRONT_PAGES and last BACK_PAGES of the document.
    If the document is short enough that front and back overlap, the full text
    is returned without duplication.

    First pages  → capture: caption, title, court, parties, docket, judge
    Last pages   → capture: signature block with "Dated:" (filing date)
    """
    from pipeline.preprocess import lines_to_text

    all_lines = zones.all_lines
    if not all_lines:
        return zones.full_text_clean

    page_nums = sorted({il.page_num for il in all_lines})
    front_pages = set(page_nums[:FRONT_PAGES])
    back_pages  = set(page_nums[-BACK_PAGES:])
    keep_pages  = front_pages | back_pages

    front_lines = [il for il in all_lines if il.page_num in front_pages]
    back_lines  = [il for il in all_lines if il.page_num in back_pages
                   and il.page_num not in front_pages]   # no duplication

    parts: list[str] = []
    if front_lines:
        parts.append(lines_to_text(front_lines))
    if back_lines:
        # separator so the model knows there is a gap
        parts.append("[...middle pages omitted...]")
        parts.append(lines_to_text(back_lines))

    return "\n".join(parts).replace("\x00", " ")


def _build_prompt(zones) -> str:
    """Return the user-turn text (system + hints + skeleton + page window)."""
    excerpt = _extract_page_window(zones)
    return (
        f"{_SYSTEM}\n\n"
        f"{_FIELD_HINTS}\n\n"
        f"Fill this template:\n{_JSON_SKELETON}\n\n"
        f"DOCUMENT (first {FRONT_PAGES} and last {BACK_PAGES} pages):\n{excerpt}"
    )


def _label_to_json(record: dict) -> str:
    """Compact JSON string of the ground-truth fields (model target)."""
    out = {
        "title":          record.get("title"),
        "document_types": record.get("document_types") or [],
        "nuid":           record.get("nuid"),
        "court_name":     record.get("court_name"),
        "court_location": record.get("court_location"),
        "judge_name":     record.get("judge_name"),
        "filing_date":    record.get("filing_date"),
        "parties":        record.get("parties") or {
            "plaintiffs": [], "defendants": [], "petitioners": [], "respondents": []
        },
        "clauses":        record.get("clauses") or [],
    }
    return json.dumps(out, ensure_ascii=False)


def _format_chat(zones, label_json: str, tokenizer) -> str:
    """
    Format a single training example using Gemma's chat template.

    The resulting string looks like:
        <bos><start_of_turn>user
        {prompt}
        <end_of_turn>
        <start_of_turn>model
        {json}
        <end_of_turn>
    """
    messages = [
        {"role": "user",  "content": _build_prompt(zones)},
        {"role": "model", "content": label_json},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(tokenizer):
    """
    Load (document_text, ground_truth_json) pairs from DOC_DIR + LABELS_JSON,
    format them with the chat template, and return a HuggingFace Dataset.
    """
    from datasets import Dataset

    # --- load pipeline so we can extract text from any supported format ---
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.pdf_loader import load_document
    from pipeline.preprocess import preprocess

    with open(LABELS_JSON) as f:
        labels: list[dict] = json.load(f)

    rows: list[dict] = []
    skipped = 0

    for record in labels:
        fp = Path(record.get("file_path", ""))
        if not fp.is_absolute():
            fp = DOC_DIR / fp
        if not fp.exists():
            skipped += 1
            continue

        try:
            doc   = load_document(str(fp))
            zones = preprocess(doc)
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        label_json = _label_to_json(record)
        text = _format_chat(zones, label_json, tokenizer)
        rows.append({"text": text})

    print(f"Dataset: {len(rows)} examples loaded, {skipped} skipped.")
    if not rows:
        raise ValueError("No training examples — check DOC_DIR and LABELS_JSON paths.")

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = build_dataset(tokenizer)

    train_cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # effective batch = 16
        gradient_checkpointing=True,     # trade compute for memory
        learning_rate=2e-4,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        use_cpu=True,
        max_length=MAX_SEQ_LEN,
        packing=False,                   # packing needs flash-attn; disable to avoid OOM spikes
        loss_type="nll",                 # plain NLL loss — skips per-token entropy that spikes RAM
        dataset_text_field="text",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=train_cfg,
    )

    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\nSaved LoRA adapter to: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Inference with trained adapter
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer():
    """Load the base model + LoRA adapter and tokenizer. Call once, reuse."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not (OUT_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No trained adapter found at {OUT_DIR}. Run training first."
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(OUT_DIR))
    model.eval()
    return model, tok


def _infer_one(doc_path: str, model, tok) -> dict:
    """
    Run the trained adapter on a single document file.
    Returns the parsed metadata dict (or raw string on parse failure).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.pdf_loader import load_document
    from pipeline.preprocess import preprocess

    doc   = load_document(doc_path)
    zones = preprocess(doc)

    messages = [{"role": "user", "content": _build_prompt(zones)}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    try:
        import json_repair
        return json_repair.loads(response)
    except Exception:
        return {"_raw": response}


def infer(doc_path: str):
    """Single-document inference — prints result to stdout."""
    model, tok = _load_model_and_tokenizer()
    result = _infer_one(doc_path, model, tok)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def infer_folder(folder_path: str):
    """
    Run inference on every supported document in folder_path.
    Results are saved to OUT_DIR/inference_results.json and also
    printed as JSONL to stdout.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.pdf_loader import SUPPORTED_EXTENSIONS

    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    docs = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not docs:
        print(f"No supported documents found in {folder_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(docs)} document(s). Loading model…")
    model, tok = _load_model_and_tokenizer()

    results: list[dict] = []
    for i, doc_path in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc_path.name} … ", end="", flush=True)
        try:
            meta = _infer_one(str(doc_path), model, tok)
            record = {"file_path": str(doc_path), **meta}
            print("done")
        except Exception as e:
            record = {"file_path": str(doc_path), "_error": str(e)}
            print(f"ERROR: {e}", file=sys.stderr)

        results.append(record)
        print(json.dumps(record, ensure_ascii=False))   # JSONL to stdout

    out_file = OUT_DIR / "inference_results.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} result(s) to {out_file}")


# ---------------------------------------------------------------------------
# TensorBoard visualizations → static matplotlib PNGs
# ---------------------------------------------------------------------------

def visualize(viz_out: str | None = None):
    """
    Read TensorBoard event files from OUT_DIR and save matplotlib plots for:
      - Training loss
      - Learning rate schedule
      - Token accuracy (if logged)
      - Gradient norm

    Args:
        viz_out: directory to save PNGs. Defaults to OUT_DIR/plots/.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend; safe on headless servers
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency for visualization: {e}")
        print("  pip install tensorboard matplotlib")
        return

    out_dir = Path(viz_out) if viz_out else OUT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate event files — HF Trainer nests them under OUT_DIR directly
    ea = EventAccumulator(str(OUT_DIR))
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    if not available:
        print(f"No scalar events found in {OUT_DIR}. Have you run training yet?")
        print(f"  tensorboard --logdir {OUT_DIR}")
        return

    print(f"Available scalar tags: {available}")

    # Metric groups to plot — (tag_substring, friendly_label, plot_title, filename)
    _METRIC_SPECS = [
        ("loss",                "Loss",           "Training Loss",            "loss.png"),
        ("learning_rate",       "LR",             "Learning Rate Schedule",   "lr.png"),
        ("mean_token_accuracy", "Token Acc.",     "Mean Token Accuracy",      "token_accuracy.png"),
        ("grad_norm",           "Grad Norm",      "Gradient Norm",            "grad_norm.png"),
        ("entropy",             "Entropy",        "Per-token Entropy",        "entropy.png"),
    ]

    plotted: list[str] = []

    for tag_sub, ylabel, title, fname in _METRIC_SPECS:
        matched = [t for t in available if tag_sub in t]
        if not matched:
            continue

        fig, ax = plt.subplots(figsize=(9, 4))
        for tag in matched:
            events = ea.Scalars(tag)
            steps  = [e.step  for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, linewidth=1.8, label=tag.split("/")[-1])

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        save_path = out_dir / fname
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        plotted.append(str(save_path))

    # Combined summary figure: loss + lr side-by-side
    loss_tags = [t for t in available if "loss" in t]
    lr_tags   = [t for t in available if "learning_rate" in t]
    if loss_tags and lr_tags:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        for tag in loss_tags:
            ev = ea.Scalars(tag)
            ax1.plot([e.step for e in ev], [e.value for e in ev],
                     linewidth=1.8, label=tag.split("/")[-1])
        ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

        for tag in lr_tags:
            ev = ea.Scalars(tag)
            ax2.plot([e.step for e in ev], [e.value for e in ev],
                     linewidth=1.8, color="tab:orange", label=tag.split("/")[-1])
        ax2.set_xlabel("Step"); ax2.set_ylabel("Learning Rate")
        ax2.set_title("LR Schedule"); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        fig.suptitle("Gemma 3 270M LoRA — Training Summary", fontsize=12, fontweight="bold")
        fig.tight_layout()
        summary_path = out_dir / "summary.png"
        fig.savefig(summary_path, dpi=150)
        plt.close(fig)
        plotted.append(str(summary_path))

    if plotted:
        print(f"\nSaved {len(plotted)} plot(s) to {out_dir}:")
        for p in plotted:
            print(f"  {p}")
    else:
        print("No matching metrics to plot.")


# ---------------------------------------------------------------------------
# Adapter layer inspection
# ---------------------------------------------------------------------------

def inspect_layers():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_cfg = OUT_DIR / "adapter_config.json"
    if not adapter_cfg.exists():
        print(f"No trained adapter found at {OUT_DIR} — run training first.")
        return

    tok  = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(OUT_DIR))

    adapter_layers = [
        name for name, _ in model.named_modules()
        if any(k in name for k in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
    ]
    print(f"\n{len(adapter_layers)} adapted module(s):")
    for name in adapter_layers:
        print(f"  {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 270M LoRA training for metadata extraction")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--infer", metavar="DOC_PATH",
                       help="Run inference on a single document")
    group.add_argument("--infer-folder", metavar="FOLDER_PATH",
                       help="Run inference on all documents in a folder; "
                            f"results saved to {OUT_DIR}/inference_results.json")
    group.add_argument("--inspect-layers", action="store_true",
                       help="Print all LoRA-adapted module names")
    group.add_argument("--visualize", action="store_true",
                       help="Generate matplotlib plots from TensorBoard event files")
    parser.add_argument("--viz-out", metavar="DIR", default=None,
                        help="Directory to save plots (default: <OUT_DIR>/plots/)")
    args = parser.parse_args()

    if args.infer:
        infer(args.infer)
    elif args.infer_folder:
        infer_folder(args.infer_folder)
    elif args.inspect_layers:
        inspect_layers()
    elif args.visualize:
        visualize(args.viz_out)
    else:
        train()
