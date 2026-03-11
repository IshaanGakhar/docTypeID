"""
Attention-based metadata extraction using specialised BERT heads.

Loads the head rankings produced by ``head_analysis.py`` and uses only the
top-K heads per entity type to extract metadata from legal documents.

Usage::

    # Single document
    python -m bert_explore.extract_with_heads --doc /path/to/doc.pdf

    # Folder of documents
    python -m bert_explore.extract_with_heads --folder /path/to/docs/ --output results.json

    # Custom heads config + model
    python -m bert_explore.extract_with_heads --doc doc.pdf \\
        --heads-config bert_explore/heads.json \\
        --model nlpaueb/legal-bert-base-uncased
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
    _merge_wordpiece,
)

HEADS_JSON = Path(__file__).resolve().parent / "heads.json"

EXTRACTION_QUERIES: dict[str, list[str]] = {
    "judge_name":      ["judge name", "who is the judge?", "honorable judge"],
    "filing_date":     ["filing date", "when was it filed?", "date filed"],
    "nuid":            ["case number", "docket number", "cause number"],
    "court_name":      ["court name", "which court?", "name of the court"],
    "court_location":  ["court location", "district of", "county of"],
    "plaintiffs":      ["name of plaintiff", "who is the plaintiff?"],
    "defendants":      ["name of defendant", "who is the defendant?"],
}

TOP_K = 10


def _load_heads_config(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load top-K (layer, head) pairs per entity type from heads.json.

    Maps extraction field names to the entity type used during analysis
    (name → judge/party, date → filing_date, number → nuid).
    """
    if not path.exists():
        return {}

    raw = json.loads(path.read_text())

    field_to_entity = {
        "judge_name":     "name",
        "plaintiffs":     "name",
        "defendants":     "name",
        "filing_date":    "date",
        "nuid":           "number",
        "court_name":     "name",
        "court_location": "name",
    }

    result: dict[str, list[tuple[int, int]]] = {}
    for field, et in field_to_entity.items():
        heads_list = raw.get(et, [])[:TOP_K]
        result[field] = [(h["layer"], h["head"]) for h in heads_list]

    return result


def _load_document_text(path: str) -> str:
    """Load first few pages of text from a document."""
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
        return p.read_text(errors="replace")[:8000]
    else:
        return p.read_text(errors="replace")[:8000]


def _extract_spans(
    query: str,
    doc_text: str,
    heads: list[tuple[int, int]],
    model_name: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Run a query against a document using specific heads.

    Returns the top-N document words by attention weight, with scores.
    """
    if not heads:
        from bert_explore.attention_probe import compute_cross_attention
        _, words_d, _, raw = compute_cross_attention(query, doc_text, model_name)
        raw_arr = np.array(raw)
        col_sums = raw_arr.sum(axis=0)
        ranked = sorted(zip(words_d, col_sums), key=lambda x: -x[1])
        return [(w, float(s)) for w, s in ranked[:top_n]]

    full = compute_full_attention(query, doc_text, model_name)
    attn = full["attn"]
    q_indices = list(range(full["q_start"], full["q_end"]))
    d_indices = list(range(full["d_start"], full["d_end"]))

    if not q_indices or not d_indices:
        return []

    head_attns = []
    for l, h in heads:
        if l < attn.shape[0] and h < attn.shape[1]:
            head_attns.append(attn[l, h])

    if not head_attns:
        return []

    avg_attn = np.mean(head_attns, axis=0)
    cross = avg_attn[q_indices][:, d_indices]

    q_toks = [full["all_tokens"][i] for i in q_indices]
    cross = _merge_wordpiece(q_toks, cross, axis=0)[1]
    d_toks = [full["all_tokens"][i] for i in d_indices]
    words_d, cross = _merge_wordpiece(d_toks, cross, axis=1)

    col_sums = cross.sum(axis=0)
    ranked = sorted(zip(words_d, col_sums), key=lambda x: -x[1])
    return [(w, float(s)) for w, s in ranked[:top_n]]


def _find_contiguous_spans(
    words: list[tuple[str, float]],
    doc_text: str,
    max_gap: int = 3,
) -> list[str]:
    """Given top attended words, find contiguous spans in the original text."""
    word_set = {w.lower() for w, _ in words}
    tokens = doc_text.split()
    runs: list[list[str]] = []
    current_run: list[str] = []
    gap = 0

    for tok in tokens:
        clean = re.sub(r"[^\w]", "", tok).lower()
        if clean in word_set:
            if gap > 0 and gap <= max_gap and current_run:
                current_run.append(tok)
            else:
                if current_run:
                    runs.append(current_run)
                current_run = [tok]
            gap = 0
        else:
            gap += 1
            if gap <= max_gap and current_run:
                current_run.append(tok)

    if current_run:
        runs.append(current_run)

    runs.sort(key=len, reverse=True)
    return [" ".join(r) for r in runs if len(r) >= 1]


_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]

_DOCKET_RE = re.compile(
    r"\b\d{1,2}:\d{2}-[a-z]{2,3}-\d{3,6}\b"
    r"|\b\d{2}[a-z]{2,3}\d{3,}\b"
    r"|\bNo\.\s*[\w\-:]+",
    re.IGNORECASE,
)


def _post_process_date(spans: list[str], doc_text: str) -> Optional[str]:
    """Extract the best date string from attended spans or nearby context."""
    for span in spans:
        for pat in _DATE_PATTERNS:
            m = pat.search(span)
            if m:
                return m.group()

    for pat in _DATE_PATTERNS:
        m = pat.search(doc_text[:2000])
        if m:
            return m.group()
    return None


def _post_process_docket(spans: list[str], doc_text: str) -> Optional[str]:
    """Extract the best docket/case number."""
    for span in spans:
        m = _DOCKET_RE.search(span)
        if m:
            return m.group()

    m = _DOCKET_RE.search(doc_text[:2000])
    if m:
        return m.group()
    return None


def _post_process_name(spans: list[str]) -> Optional[str]:
    """Extract a plausible name from attended spans."""
    for span in spans:
        clean = re.sub(r"\s+", " ", span).strip()
        clean = re.sub(r"^(Hon\.?|Honorable|Judge|Justice)\s+", "", clean, flags=re.I)
        words = clean.split()
        name_words = [w for w in words if w[0:1].isupper() and len(w) > 1]
        if 2 <= len(name_words) <= 5:
            return " ".join(name_words)
    return None


def extract_metadata(
    doc_path: str,
    heads_config: Optional[dict[str, list[tuple[int, int]]]] = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Extract metadata from a document using attention-based extraction."""
    doc_text = _load_document_text(doc_path)

    if heads_config is None:
        heads_config = _load_heads_config(HEADS_JSON)

    _, tok = get_model_and_tok(model_name)
    doc_ids = tok(doc_text, add_special_tokens=False)["input_ids"]
    if len(doc_ids) > 450:
        doc_ids = doc_ids[:450]
    truncated = tok.decode(doc_ids, skip_special_tokens=True)

    result: dict = {
        "file_path": str(doc_path),
        "title": None,
        "document_types": [],
        "nuid": None,
        "court_name": None,
        "court_location": None,
        "judge_name": None,
        "filing_date": None,
        "parties": {"plaintiffs": [], "defendants": []},
        "attention_evidence": {},
    }

    for field, queries in EXTRACTION_QUERIES.items():
        heads = heads_config.get(field, [])
        all_spans: list[tuple[str, float]] = []

        for query in queries:
            spans = _extract_spans(query, truncated, heads, model_name, top_n=8)
            all_spans.extend(spans)

        if not all_spans:
            continue

        score_map: dict[str, float] = {}
        for w, s in all_spans:
            score_map[w] = score_map.get(w, 0.0) + s

        top_words = sorted(score_map.items(), key=lambda x: -x[1])[:10]
        contiguous = _find_contiguous_spans(top_words, doc_text)

        result["attention_evidence"][field] = {
            "top_words": [(w, round(s, 4)) for w, s in top_words[:5]],
            "spans": contiguous[:3],
        }

        if field == "judge_name":
            result["judge_name"] = _post_process_name(contiguous)
        elif field == "filing_date":
            result["filing_date"] = _post_process_date(contiguous, doc_text)
        elif field == "nuid":
            result["nuid"] = _post_process_docket(contiguous, doc_text)
        elif field == "court_name":
            if contiguous:
                result["court_name"] = contiguous[0]
        elif field == "court_location":
            if contiguous:
                result["court_location"] = contiguous[0]
        elif field == "plaintiffs":
            names = [_post_process_name([s]) for s in contiguous[:5]]
            result["parties"]["plaintiffs"] = [n for n in names if n]
        elif field == "defendants":
            names = [_post_process_name([s]) for s in contiguous[:5]]
            result["parties"]["defendants"] = [n for n in names if n]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from legal documents using specialised BERT heads.",
    )
    parser.add_argument("--doc", help="Path to a single document")
    parser.add_argument("--folder", help="Folder of documents to process")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument(
        "--heads-config", default=str(HEADS_JSON),
        help="Path to heads.json from head_analysis.py",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    heads_config = _load_heads_config(Path(args.heads_config))
    if not heads_config:
        print(
            "Warning: no heads.json found. Run head_analysis.py first for best results.\n"
            "Falling back to all-head averaging.",
            file=sys.stderr,
        )

    results: list[dict] = []

    if args.doc:
        print(f"Processing: {args.doc}", file=sys.stderr)
        r = extract_metadata(args.doc, heads_config, args.model)
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"ERROR: not a directory: {folder}", file=sys.stderr)
            sys.exit(1)

        exts = {".pdf", ".docx", ".doc", ".txt"}
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in exts)
        total = len(files)
        print(f"Found {total} documents in {folder}", file=sys.stderr)

        for i, fp in enumerate(files, 1):
            print(f"  [{i}/{total}] {fp.name}", file=sys.stderr)
            try:
                r = extract_metadata(str(fp), heads_config, args.model)
                results.append(r)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                results.append({"file_path": str(fp), "error": str(e)})

        out_path = Path(args.output) if args.output else folder / "bert_results.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nOutput → {out_path}", file=sys.stderr)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
