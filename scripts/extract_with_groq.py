"""
Batch metadata extraction using Groq LLM (gpt-oss-120b).

Processes every supported document in a folder, extracts structured metadata
via the Groq API, and writes a ground_truth.json ready for train_lora.py.

Usage:
    python scripts/extract_with_groq.py --folder /path/to/docs --output data/groq_labels
    python scripts/extract_with_groq.py --folder /path/to/docs --output data/groq_labels --resume
    python scripts/extract_with_groq.py --folder /path/to/docs --output data/groq_labels --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.pdf_loader import load_document
from pipeline.preprocess import preprocess, lines_to_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL       = "openai/gpt-oss-120b"
FRONT_PAGES = 2
BACK_PAGES  = 2
MAX_TOKENS  = 1024
DELAY_S     = 0.5       # between sequential requests
SUPPORTED   = {".pdf", ".docx", ".doc", ".txt"}

# ---------------------------------------------------------------------------
# Prompt (same as train_lora.py for consistency)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a legal document metadata extractor.
Read the document excerpt and fill in the JSON template exactly.
Output ONLY valid JSON — no explanation, no prose, no markdown fences.\
"""

_SKELETON = """\
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

_HINTS = """\
Field definitions:
- title: the document's own title in ALL CAPS (e.g. "MOTION TO DISMISS"), or null
- document_types: list of specific types, e.g. ["Motion to Dismiss", "Memorandum"]
- nuid: normalised docket/case number, e.g. "1:23-CV-04521-RJS", or null
- court_name: full court name, e.g. "UNITED STATES DISTRICT COURT", or null
- court_location: district or county, e.g. "Southern District of New York", or null
- judge_name: presiding judge without "Hon.", e.g. "Robert J. Sullivan", or null
- filing_date: ISO-8601 date from the "Dated:" signature line, e.g. "2023-09-28", or null
- parties.plaintiffs / defendants: names from the caption above/below "v.", or []
- clauses: US Constitutional provisions cited, e.g. ["Fourth Amendment"], or []\
"""


# ---------------------------------------------------------------------------
# Text extraction — first N + last N pages
# ---------------------------------------------------------------------------

def _extract_page_window(zones) -> str:
    all_lines = zones.all_lines
    if not all_lines:
        return zones.full_text_clean

    page_nums   = sorted({il.page_num for il in all_lines})
    front_pages = set(page_nums[:FRONT_PAGES])
    back_pages  = set(page_nums[-BACK_PAGES:])

    front_lines = [il for il in all_lines if il.page_num in front_pages]
    back_lines  = [il for il in all_lines if il.page_num in back_pages
                   and il.page_num not in front_pages]

    parts: list[str] = []
    if front_lines:
        parts.append(lines_to_text(front_lines))
    if back_lines:
        parts.append("[...middle pages omitted...]")
        parts.append(lines_to_text(back_lines))

    return "\n".join(parts).replace("\x00", " ")


def _build_prompt(excerpt: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"{_HINTS}\n\n"
        f"Fill this template:\n{_SKELETON}\n\n"
        f"DOCUMENT (first {FRONT_PAGES} and last {BACK_PAGES} pages):\n{excerpt}"
    )


# ---------------------------------------------------------------------------
# Single-document extraction
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | None:
    """Extract and parse the first JSON object from the model response."""
    import re
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return None
    blob = m.group()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            import json_repair
            return json_repair.loads(blob)
        except Exception:
            return None


def extract_one(fp: Path, client: Groq) -> dict | None:
    """
    Load a document, call Groq, return the parsed metadata dict
    (with file_path injected) or None on failure.
    """
    try:
        doc   = load_document(str(fp))
        zones = preprocess(doc)
        excerpt = _extract_page_window(zones)
    except Exception as e:
        print(f"  [load-err] {fp.name}: {e}", file=sys.stderr)
        return None

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": _build_prompt(excerpt)},
            ],
            temperature=0.1,    # low temp for deterministic extraction
            max_tokens=MAX_TOKENS,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [api-err]  {fp.name}: {e}", file=sys.stderr)
        return None

    result = _parse_json(raw)
    if result is None:
        print(f"  [json-err] {fp.name}: could not parse response", file=sys.stderr)
        return None

    result["file_path"] = str(fp.resolve())
    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def discover_docs(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def run(folder: Path, out_dir: Path, resume: bool, workers: int) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found — set it in .env")

    client = Groq(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_file = out_dir / "ground_truth.json"
    done_paths: set[str] = set()
    existing: list[dict] = []

    if resume and gt_file.exists():
        with open(gt_file) as f:
            existing = json.load(f)
        done_paths = {r["file_path"] for r in existing}
        print(f"Resuming — {len(done_paths)} already done.")

    docs = discover_docs(folder)
    pending = [p for p in docs if str(p.resolve()) not in done_paths]
    print(f"Found {len(docs)} docs, {len(pending)} to process.")

    results: list[dict] = list(existing)
    succeeded = failed = 0

    if workers > 1:
        # Parallel — mind your Groq rate limit
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, fp, client): fp for fp in pending}
            for fut in as_completed(futures):
                fp = futures[fut]
                r  = fut.result()
                if r:
                    results.append(r)
                    succeeded += 1
                    print(f"  OK   {fp.name}")
                else:
                    failed += 1
                    print(f"  FAIL {fp.name}")
                # checkpoint after every doc
                with open(gt_file, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        for fp in pending:
            print(f"  {fp.name} ...", end=" ", flush=True)
            r = extract_one(fp, client)
            if r:
                results.append(r)
                succeeded += 1
                print("OK")
            else:
                failed += 1
                print("FAIL")
            # checkpoint
            with open(gt_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            time.sleep(DELAY_S)

    print(f"\nDone: {succeeded} extracted, {failed} failed.")
    print(f"Labels written to: {gt_file}  ({len(results)} total records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch metadata extraction via Groq LLM"
    )
    parser.add_argument("--folder",  type=Path, required=True,
                        help="Folder containing documents (.pdf/.docx/.doc/.txt)")
    parser.add_argument("--output",  type=Path, default=Path("data/groq_labels"),
                        help="Output folder for ground_truth.json (default: data/groq_labels)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip docs already present in ground_truth.json")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel threads (default 1; increase carefully re: rate limits)")
    parser.add_argument("--model",   default=MODEL,
                        help=f"Groq model name (default: {MODEL})")
    args = parser.parse_args()

    MODEL = args.model
    run(folder=args.folder, out_dir=args.output,
        resume=args.resume, workers=args.workers)
