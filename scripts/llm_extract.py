"""
LLM-based metadata extraction using Gemma 3 270M.

Calls google/gemma-3-270m-it locally to extract structured JSON metadata
from a single legal document. Runs independently of the rule-based pipeline.

Usage:
    python scripts/llm_extract.py /path/to/doc.pdf
    python scripts/llm_extract.py /path/to/doc.pdf --model google/gemma-3-270m-it
    python scripts/llm_extract.py /path/to/doc.pdf --max-chars 6000 --output out.json
    python scripts/llm_extract.py /path/to/doc.pdf --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a legal document metadata extractor. \
Read the document and fill in the JSON template below. \
Output ONLY the completed JSON — no explanation, no prose, no markdown.\
Do not use placeholders, use the most relevant information from the document.\
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
  "parties": {
    "plaintiffs": [],
    "defendants": [],
    "petitioners": [],
    "respondents": []
  },
  "clauses": []
}\
"""

_FIELD_HINTS = """\
Field definitions:
- title: the document's own title (e.g. "MOTION TO DISMISS", "NOTICE OF MOTION"), or null
- document_types: list from [Motion, Order, Complaint, Petition, Answer, Brief, Notice, Memorandum, Stipulation, Judgment, Subpoena, Affidavit, Declaration, Transcript, Summons]
- nuid: docket/case number string (e.g. "18-cv-1208", "Case No. 2:19-cv-04315"), or null
- court_name: full court name (e.g. "United States District Court", "Superior Court of California"), or null
- court_location: district or county (e.g. "Southern District of California", "County of Los Angeles"), or null
- judge_name: presiding judge's full name (e.g. "Hon. Thomas E. Kuhnle"), or null
- filing_date: date as YYYY-MM-DD, or null
- parties: look in the case caption at the top — the names listed above "Plaintiff", "Defendant", etc.
    plaintiffs: list of plaintiff/petitioner names found before "v." or "vs."
    defendants: list of defendant/respondent names found after "v." or "vs."
    petitioners: list of petitioner names (if a petition-style case)
    respondents: list of respondent names
- clauses: list of {heading, clause_type} for each major numbered section or heading in the document\
"""


def build_prompt(doc_text: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"DOCUMENT TEXT (first page):\n{doc_text}\n\n"
        f"{_FIELD_HINTS}\n\n"
        f"Based ONLY on the document text above, fill every field you can find. "
        f"Use null only if the information is genuinely absent.\n\n"
        f"Filled JSON:\n{{"
    )


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

# Canonical field names and all aliases the model might generate
_KEY_ALIASES: dict[str, list[str]] = {
    "title":          ["title", "document title", "doc_title", "Title", "TITLE"],
    "document_types": ["document_types", "document_dtypes", "document types",
                       "Document Types", "DOCTYPE", "doc_type", "doctype", "type"],
    "nuid":           ["nuid", "Nuid", "NUID", "case_number", "docket", "docket_number",
                       "case number", "Case Number"],
    "court_name":     ["court_name", "court_Name", "COURT_NAME", "COUR_NAME",
                       "court name", "Court Name", "courthouse", "Courthouse Name"],
    "court_location": ["court_location", "court_Location", "Court Location",
                       "COURT_LOCATION", "location", "jurisdiction", "district"],
    "judge_name":     ["judge_name", "judge_Name", "Judge_Name", "JUDGE_NAME",
                       "judge", "Judge", "presiding_judge", "Honorable", "Honorable Judge",
                       "Honorable Judge Name", "Honorable Judge's Name", "Honorable Judge's Name's",
                       "Honorable Judge's Name's Name's", "Hon."],
    "filing_date":    ["filing_date", "Filings_Date", "filing date", "Filing Date",
                       "file_format", "date_filed", "Date Filed", "filed"],
    "parties":        ["parties", "Parties", "PARTIES", "party"],
    "clauses":        ["clauses", "Clauses", "CLAUSES", "sections", "clause"],
}

_CANONICAL_KEYS = set(_KEY_ALIASES.keys())

def _to_iso_date(val: str | None) -> str | None:
    """Try to parse a date string and return YYYY-MM-DD, or return the original."""
    if not val or not isinstance(val, str):
        return val
    try:
        from dateutil import parser as du_parser
        return du_parser.parse(val, default=None).strftime("%Y-%m-%d")
    except Exception:
        return val


def _normalise_keys(obj: dict) -> dict:
    """Map whatever key names the model produced to our canonical schema keys,
    drop all unrecognised keys, and apply value normalisation."""
    # Build reverse lookup: alias -> canonical
    alias_to_canon: dict[str, str] = {}
    for canon, aliases in _KEY_ALIASES.items():
        for a in aliases:
            alias_to_canon[a.lower()] = canon

    out: dict = {}
    for k, v in obj.items():
        canon = alias_to_canon.get(k.strip().lower())
        if canon is not None:          # only keep recognised keys
            out.setdefault(canon, v)   # first match wins

    # Ensure all canonical keys are present with defaults
    defaults: dict = {
        "title": None, "document_types": [], "nuid": None,
        "court_name": None, "court_location": None, "judge_name": None,
        "filing_date": None,
        "parties": {"plaintiffs": [], "defendants": [], "petitioners": [], "respondents": []},
        "clauses": [],
    }
    for key, default in defaults.items():
        out.setdefault(key, default)

    # Normalise filing_date to ISO-8601
    out["filing_date"] = _to_iso_date(out.get("filing_date"))

    # Normalise parties sub-keys
    p_alias = {
        "plaintiff": "plaintiffs", "plaintiffs": "plaintiffs",
        "defendant": "defendants", "defendants": "defendants",
        "petitioner": "petitioners", "petitioners": "petitioners",
        "respondent": "respondents", "respondents": "respondents",
        "defendent": "defendants", "Defendent": "defendants",
        "Plaintiff": "plaintiffs",
    }
    parties = out.get("parties", {})
    if isinstance(parties, list):
        out["parties"] = defaults["parties"]
    elif isinstance(parties, dict):
        party_defaults = {"plaintiffs": [], "defendants": [], "petitioners": [], "respondents": []}
        normalised_parties: dict = {**party_defaults}
        for pk, pv in parties.items():
            canon_pk = p_alias.get(pk, p_alias.get(pk.lower()))
            if canon_pk and canon_pk in normalised_parties:
                normalised_parties[canon_pk] = pv if isinstance(pv, list) else []
        out["parties"] = normalised_parties

    # Return only canonical keys (excludes junk keys json-repair may have kept)
    return {k: out[k] for k in defaults}


def extract_json(raw: str) -> dict:
    """
    Pull the first valid JSON object out of the model's raw output.
    Tries four approaches in order:
      1. Direct parse
      2. Extract from markdown fences
      3. Find the first balanced { … } block
      4. json-repair (tolerates syntax errors from small models)
    After parsing, normalise field names to the canonical schema.
    """
    # The prompt ends with "{" which we injected — prepend it so the raw output
    # forms a complete JSON object.
    text = ("{" + raw).strip() if not raw.strip().startswith("{") else raw.strip()

    def _try_parse(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return _normalise_keys(obj)
        except json.JSONDecodeError:
            pass
        return None

    # 1. Direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Markdown fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        result = _try_parse(fence.group(1))
        if result is not None:
            return result

    # 3. First balanced brace block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start : i + 1])
                    if result is not None:
                        return result
                    break

    # 4. json-repair — tolerates missing quotes, trailing commas, mixed quotes, etc.
    try:
        from json_repair import repair_json
        candidate = text
        # Prefer the fenced block if present
        if fence:
            candidate = fence.group(1)
        elif start != -1:
            candidate = text[start:]
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict):
            return _normalise_keys(repaired)
        # repair_json may return a string if it produced a JSON string
        if isinstance(repaired, str):
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return _normalise_keys(obj)
    except Exception:
        pass

    raise ValueError(f"No valid JSON found in model output:\n{raw[:500]}")


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str, hf_token: str | None = None):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("ERROR: transformers and torch are required. Install with:\n"
              "  pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    kwargs = {"token": hf_token} if hf_token else {}

    print(f"Loading {model_id} on {device} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype="auto",
        **kwargs,
    )
    model.eval()
    print("Model loaded.", file=sys.stderr)
    return tokenizer, model, device


def run_inference(
    tokenizer,
    model,
    device: str,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    import torch

    # Gemma 3 uses the standard chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        result = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # Newer transformers returns a BatchEncoding dict; older versions return a plain tensor
        if isinstance(result, dict):
            inputs = {k: v.to(device) for k, v in result.items()}
            prompt_len = inputs["input_ids"].shape[-1]
        else:
            inputs = {"input_ids": result.to(device)}
            prompt_len = result.shape[-1]
    except Exception:
        # Fallback: plain tokenization if chat template unavailable
        enc = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in enc.items()}
        prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,   # penalise repeated tokens to break loops
            no_repeat_ngram_size=6,   # forbid repeating any 6-gram
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Document loading (reuses pipeline loader)
# ---------------------------------------------------------------------------

def load_doc_text(path: Path, max_chars: int) -> str:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from pipeline.pdf_loader import load_document
        doc = load_document(path)
        if doc.skipped:
            print(f"WARNING: document skipped — {doc.skip_reason}", file=sys.stderr)
            return ""
        # Prefer first-page text — it contains the caption, title, court, parties, and date.
        # Fall back to full_text if first_page_text is too short.
        first = (doc.first_page_text or "").strip()
        if len(first) >= 200:
            text = first
            print(f"Using first-page text ({len(first)} chars)", file=sys.stderr)
        else:
            text = doc.full_text
            print(f"First-page text too short ({len(first)} chars), using full text",
                  file=sys.stderr)
        return text[:max_chars]
    except Exception as exc:
        # Plain-text fallback
        print(f"WARNING: pipeline loader failed ({exc}), falling back to plain read",
              file=sys.stderr)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(errors="replace")[:max_chars]
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract legal metadata with Gemma 3 270M")
    p.add_argument("document", type=Path, help="Path to PDF, DOCX, TXT, or DOC file")
    p.add_argument(
        "--model", default="google/gemma-3-1b-it",
        help="HuggingFace model ID (default: google/gemma-3-270m-it)",
    )
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (default: auto-detect)",
    )
    p.add_argument(
        "--max-chars", type=int, default=4000,
        help="Max document characters sent to the model (default: 4000). "
             "Smaller values keep the model focused on the caption/header.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens the model may generate (default: 512)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON result to this file (default: print to stdout)",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="HuggingFace access token (needed for gated models like Gemma). "
             "Alternative: run `huggingface-cli login` once beforehand.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.document.exists():
        print(f"ERROR: file not found: {args.document}", file=sys.stderr)
        sys.exit(1)

    # 1. Load document text
    print(f"Loading document: {args.document}", file=sys.stderr)
    doc_text = load_doc_text(args.document, args.max_chars)
    if not doc_text.strip():
        print("ERROR: could not extract text from document.", file=sys.stderr)
        sys.exit(1)
    print(f"Document text: {len(doc_text)} chars", file=sys.stderr)

    # 2. Build prompt
    prompt = build_prompt(doc_text)

    # 3. Load model and run
    tokenizer, model, device = load_model(args.model, args.device, args.hf_token)
    print("Running inference …", file=sys.stderr)
    raw_output = run_inference(tokenizer, model, device, prompt, args.max_new_tokens)
    print(f"Raw output ({len(raw_output)} chars):\n{raw_output[:300]}\n…",
          file=sys.stderr)

    # 4. Parse JSON
    try:
        result = extract_json(raw_output)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        result = {"error": "json_parse_failed", "raw_output": raw_output}

    # 5. Attach metadata
    result["_meta"] = {
        "source_file": str(args.document),
        "model": args.model,
        "doc_chars_used": len(doc_text),
    }

    # 6. Output
    out_str = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(out_str)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
