"""
Probe every BERT head to discover which (layer, head) pairs specialise in
names, dates, or numbers.

Usage::

    # Analyse against a single document string
    python -m bert_explore.head_analysis --text "FILED: NEW YORK COUNTY CLERK 03/22/2021"

    # Analyse against a document file
    python -m bert_explore.head_analysis --doc /path/to/doc.pdf

    # Use legal-bert
    python -m bert_explore.head_analysis --model nlpaueb/legal-bert-base-uncased --doc doc.pdf

Outputs:
    bert_explore/plots/heatmap_<entity_type>.png   — 12×12 layer-vs-head heatmaps
    bert_explore/plots/top_heads.png               — bar chart of top-10 heads per type
    bert_explore/heads.json                        — serialised ranked head list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from bert_explore.attention_probe import (
    DEFAULT_MODEL,
    compute_full_attention,
    get_model_and_tok,
    model_info,
    _merge_wordpiece,
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
HEADS_JSON = Path(__file__).resolve().parent / "heads.json"

TYPED_QUERIES: dict[str, list[str]] = {
    "name": [
        "who is the judge?",
        "name of plaintiff",
        "name of defendant",
        "judge name",
    ],
    "date": [
        "when was it filed?",
        "filing date",
        "date of order",
        "effective date",
    ],
    "number": [
        "case number",
        "docket number",
        "cause number",
        "index number",
    ],
}

# Regex matchers for each entity type applied to document tokens
_DATE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"\d{2,}[-:]\w+[-:]\d+"
    r"|\d{2,}[a-z]{2}\d+"
    r"|\b\d{4,}\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def _classify_token(tok_str: str) -> set[str]:
    """Return the set of entity types a token string matches."""
    types: set[str] = set()
    if _DATE_RE.fullmatch(tok_str) or _DATE_RE.search(tok_str):
        types.add("date")
    if _NUMBER_RE.fullmatch(tok_str) or _NUMBER_RE.search(tok_str):
        types.add("number")
    if _NAME_RE.match(tok_str):
        types.add("name")
    if tok_str.isupper() and len(tok_str) > 2 and tok_str.isalpha():
        types.add("name")
    return types


def _load_document_text(path: str) -> str:
    """Load text from a PDF, DOCX, or TXT file."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(str(p))
        pages = [doc[i].get_text() for i in range(min(len(doc), 3))]
        return "\n".join(pages)
    elif ext == ".docx":
        import docx
        d = docx.Document(str(p))
        return "\n".join(para.text for para in d.paragraphs[:100])
    elif ext in (".txt", ".text"):
        return p.read_text(errors="replace")[:5000]
    else:
        return p.read_text(errors="replace")[:5000]


def analyse_heads(
    document_text: str,
    model_name: str = DEFAULT_MODEL,
    max_doc_tokens: int = 450,
) -> dict[str, list[tuple[int, int, float]]]:
    """Score every (layer, head) for each entity type.

    Returns ``{entity_type: [(layer, head, score), ...]}`` sorted descending.
    """
    info = model_info(model_name)
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]

    _, tok = get_model_and_tok(model_name)
    doc_ids = tok(document_text, add_special_tokens=False)["input_ids"]
    if len(doc_ids) > max_doc_tokens:
        doc_ids = doc_ids[:max_doc_tokens]
    truncated_doc = tok.decode(doc_ids, skip_special_tokens=True)

    doc_tokens_raw = tok.convert_ids_to_tokens(doc_ids)
    words_d, _ = _merge_wordpiece(
        doc_tokens_raw, np.zeros((len(doc_tokens_raw), 1)), axis=0
    )
    word_types: list[set[str]] = [_classify_token(w) for w in words_d]

    scores: dict[str, np.ndarray] = {
        et: np.zeros((n_layers, n_heads), dtype=np.float64)
        for et in TYPED_QUERIES
    }
    query_counts: dict[str, int] = {et: 0 for et in TYPED_QUERIES}

    for entity_type, queries in TYPED_QUERIES.items():
        target_indices = [i for i, wt in enumerate(word_types) if entity_type in wt]
        if not target_indices:
            continue

        for query in queries:
            full = compute_full_attention(query, truncated_doc, model_name)
            attn = full["attn"]  # (L, H, S, S)
            q_indices = list(range(full["q_start"], full["q_end"]))
            d_indices = list(range(full["d_start"], full["d_end"]))

            if not q_indices or not d_indices:
                continue

            for l_idx in range(n_layers):
                for h_idx in range(n_heads):
                    head_attn = attn[l_idx, h_idx]
                    cross = head_attn[q_indices][:, d_indices]

                    q_toks = [full["all_tokens"][i] for i in q_indices]
                    _, cross = _merge_wordpiece(q_toks, cross, axis=0)
                    d_toks = [full["all_tokens"][i] for i in d_indices]
                    _, cross = _merge_wordpiece(d_toks, cross, axis=1)

                    total_attn = cross.sum()
                    if total_attn == 0:
                        continue

                    target_mass = cross[:, target_indices].sum()
                    scores[entity_type][l_idx, h_idx] += target_mass / total_attn

            query_counts[entity_type] += 1

    results: dict[str, list[tuple[int, int, float]]] = {}
    for et in TYPED_QUERIES:
        if query_counts[et] > 0:
            scores[et] /= query_counts[et]

        flat: list[tuple[int, int, float]] = []
        for l in range(n_layers):
            for h in range(n_heads):
                flat.append((l, h, float(scores[et][l, h])))
        flat.sort(key=lambda x: x[2], reverse=True)
        results[et] = flat

    return results


def save_heads_json(results: dict[str, list[tuple[int, int, float]]], path: Path = HEADS_JSON):
    """Persist the top heads per entity type to JSON."""
    out: dict[str, list[dict]] = {}
    for et, ranked in results.items():
        out[et] = [
            {"layer": l, "head": h, "score": round(s, 6)}
            for l, h, s in ranked[:20]
        ]
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved head rankings → {path}", file=sys.stderr)


def plot_heatmaps(
    results: dict[str, list[tuple[int, int, float]]],
    n_layers: int = 12,
    n_heads: int = 12,
    out_dir: Path = PLOTS_DIR,
):
    """Generate a 12×12 heatmap per entity type + a combined top-10 bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for et, ranked in results.items():
        grid = np.zeros((n_layers, n_heads))
        for l, h, s in ranked:
            grid[l, h] = s

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Attention specialisation: {et}")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        plt.colorbar(im, ax=ax, label="Specialisation score")
        fig.tight_layout()
        path = out_dir / f"heatmap_{et}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}", file=sys.stderr)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    for ax, (et, ranked) in zip(axes, results.items()):
        top10 = ranked[:10]
        labels = [f"L{l}H{h}" for l, h, _ in top10]
        values = [s for _, _, s in top10]
        colors = {"name": "#4a90d9", "date": "#50b848", "number": "#e8a838"}
        ax.barh(labels[::-1], values[::-1], color=colors.get(et, "#888888"))
        ax.set_xlabel("Specialisation score")
        ax.set_title(f"Top-10 heads: {et}")

    fig.tight_layout()
    path = out_dir / "top_heads.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Probe BERT heads for entity-type specialisation.",
    )
    parser.add_argument("--text", help="Document text to analyse (inline string)")
    parser.add_argument("--doc", help="Path to a document file (PDF/DOCX/TXT)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if args.doc:
        print(f"Loading document: {args.doc}", file=sys.stderr)
        doc_text = _load_document_text(args.doc)
    elif args.text:
        doc_text = args.text
    else:
        doc_text = (
            "UNITED STATES DISTRICT COURT\n"
            "SOUTHERN DISTRICT OF NEW YORK\n"
            "Case No. 1:23-cv-00515-NRB\n"
            "MADELEINE BRAND and JOHN BOWDEN,\n"
            "Plaintiffs,\n"
            "v.\n"
            "GODIVA CHOCOLATIER, INC.,\n"
            "Defendant.\n"
            "FILED: January 19, 2023\n"
            "Hon. Naomi Reice Buchwald\n"
        )
        print("No --text or --doc provided, using built-in sample.", file=sys.stderr)

    print(f"Document length: {len(doc_text)} chars", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print("Analysing all heads...", file=sys.stderr)

    results = analyse_heads(doc_text, model_name=args.model)

    print("\n=== Top-5 heads per entity type ===", file=sys.stderr)
    for et, ranked in results.items():
        print(f"\n  {et.upper()}:", file=sys.stderr)
        for l, h, s in ranked[:5]:
            print(f"    Layer {l:2d}, Head {h:2d}: {s:.4f}", file=sys.stderr)

    save_heads_json(results)
    info = model_info(args.model)
    plot_heatmaps(results, n_layers=info["n_layers"], n_heads=info["n_heads"])

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
