"""
Legal metadata extraction using Groq gpt-oss-20b with best-of-N expert sampling.

For each document the model is called N times (default 3) with a high-expertise
legal-specialist system prompt. The "best" response is selected by picking the
candidate with the most non-null fields — the expert answer wins.

Usage:
    python scripts/infer_groq_expert.py --folder /path/to/docs
    python scripts/infer_groq_expert.py --folder /path/to/docs --output results/expert.json
    python scripts/infer_groq_expert.py --folder /path/to/docs --n 5 --workers 2 --resume
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

MODEL       = "openai/gpt-oss-20b"
FRONT_PAGES = 2
BACK_PAGES  = 2
MAX_TOKENS  = 1024
N_SAMPLES   = 3       # best-of-N: call the model this many times, pick best
TEMPERATURE = 0.3     # slightly above 0 so samples differ; expert consistency
DELAY_S     = 0.3
SUPPORTED   = {".pdf", ".docx", ".doc", ".txt"}

# ---------------------------------------------------------------------------
# Expert system prompt — stronger legal-specialist persona than extract_with_groq
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a senior US federal litigation paralegal with 20 years of experience
reading court filings, motions, orders, and transactional legal documents.

Your task is to extract structured metadata from a document excerpt with
maximum precision. Apply these expert rules:

1. TITLE  — Look for the document's own heading in ALL CAPS (e.g. "MOTION TO
   DISMISS", "MEMORANDUM OF LAW IN SUPPORT OF …"). Never use the case caption
   or party names as the title.
2. NUID   — The docket or case number, normalised (e.g. "1:23-cv-04521-NRB").
   Variants: "Case No.", "Docket No.", "No.", "Civil Action No.".
3. COURT  — Full official court name (e.g. "UNITED STATES DISTRICT COURT").
4. LOCATION — The district or county jurisdiction only (e.g. "Southern District
   of New York", "County of Los Angeles").
5. JUDGE  — Name without title prefix ("Hon.", "Judge", "Justice").
6. DATE   — ISO-8601 filing date (YYYY-MM-DD) from the "Dated:" signature block
   or filing stamp. If none, null.
7. PARTIES — Names from the case caption above/below "v." only. Do not include
   attorneys, law firms, or addresses.
8. CLAUSES — Only explicit US Constitutional citations: amendments by number/
   name (e.g. "Fourth Amendment"), named clauses ("Due Process Clause"), or
   article/section references ("Article III"). Empty list if none.
9. DOC TYPES — Be specific (prefer "Motion to Dismiss" over generic "Motion").

If a field genuinely cannot be determined from the excerpt, use null / [].
Output ONLY valid JSON. No prose, no markdown, no explanation.\
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


# ---------------------------------------------------------------------------
# Text helpers
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


def _build_user_msg(excerpt: str) -> str:
    return (
        f"Fill this JSON template using the document excerpt below.\n\n"
        f"Template:\n{_SKELETON}\n\n"
        f"DOCUMENT EXCERPT (first {FRONT_PAGES} + last {BACK_PAGES} pages):\n"
        f"{excerpt}"
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | None:
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


# ---------------------------------------------------------------------------
# Scoring — pick the "best" candidate
# ---------------------------------------------------------------------------

_SCALAR_FIELDS  = ["title", "nuid", "court_name", "court_location",
                    "judge_name", "filing_date"]
_LIST_FIELDS    = ["document_types", "clauses"]
_PARTY_ROLES    = ["plaintiffs", "defendants", "petitioners", "respondents"]

def _score(candidate: dict) -> int:
    """
    Count non-null, non-empty fields.  More filled-in = better expert answer.
    Party sub-fields and list entries each count separately so richer answers win.
    """
    score = 0
    for f in _SCALAR_FIELDS:
        if candidate.get(f) not in (None, "", []):
            score += 2                      # scalars worth 2 (more specific)
    for f in _LIST_FIELDS:
        score += len(candidate.get(f) or [])
    parties = candidate.get("parties") or {}
    for role in _PARTY_ROLES:
        score += len(parties.get(role) or [])
    return score


def _best_of(candidates: list[dict]) -> dict:
    """Return the highest-scoring candidate, merging ties by union of fields."""
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]

    ranked = sorted(candidates, key=_score, reverse=True)
    best   = ranked[0]
    best_s = _score(best)

    # Merge any ties
    merged = dict(best)
    for runner in ranked[1:]:
        if _score(runner) < best_s:
            break
        for f in _SCALAR_FIELDS:
            if merged.get(f) is None and runner.get(f) is not None:
                merged[f] = runner[f]
        for f in _LIST_FIELDS:
            combined = list({v for v in (merged.get(f) or []) + (runner.get(f) or [])})
            if combined:
                merged[f] = combined
        p_m = merged.setdefault("parties", {})
        p_r = runner.get("parties") or {}
        for role in _PARTY_ROLES:
            combined = list({v for v in (p_m.get(role) or []) + (p_r.get(role) or [])})
            if combined:
                p_m[role] = combined

    return merged


# ---------------------------------------------------------------------------
# Single-document extraction (best-of-N)
# ---------------------------------------------------------------------------

def extract_one(fp: Path, client: Groq, n: int = N_SAMPLES) -> dict | None:
    try:
        doc    = load_document(str(fp))
        zones  = preprocess(doc)
        excerpt = _extract_page_window(zones)
    except Exception as e:
        print(f"  [load-err] {fp.name}: {e}", file=sys.stderr)
        return None

    user_msg = _build_user_msg(excerpt)
    candidates: list[dict] = []

    for attempt in range(1, n + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [api-err]  {fp.name} attempt {attempt}: {e}", file=sys.stderr)
            continue

        parsed = _parse_json(raw)
        if parsed:
            candidates.append(parsed)

    if not candidates:
        print(f"  [no-parse] {fp.name}: all {n} attempts failed", file=sys.stderr)
        return None

    result = _best_of(candidates)
    result["file_path"] = str(fp.resolve())
    result["_n_samples"] = len(candidates)
    result["_best_score"] = _score(result)
    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def discover_docs(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def run(folder: Path, out_file: Path, resume: bool, workers: int, n: int) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found — add it to .env")

    client = Groq(api_key=api_key)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    done_paths: set[str] = set()
    existing:   list[dict] = []

    if resume and out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        done_paths = {r["file_path"] for r in existing}
        print(f"Resuming — {len(done_paths)} already done.")

    docs    = discover_docs(folder)
    pending = [p for p in docs if str(p.resolve()) not in done_paths]
    print(f"Found {len(docs)} doc(s), {len(pending)} pending. "
          f"Model: {MODEL}, N={n}")

    results: list[dict] = list(existing)
    ok = fail = 0

    def _process(fp: Path) -> tuple[Path, dict | None]:
        r = extract_one(fp, client, n)
        time.sleep(DELAY_S)
        return fp, r

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, fp): fp for fp in pending}
            for fut in as_completed(futures):
                fp, r = fut.result()
                if r:
                    results.append(r)
                    ok += 1
                    print(f"  OK   [{r['_best_score']:>3}pts] {fp.name}")
                else:
                    fail += 1
                    print(f"  FAIL {fp.name}")
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        for fp in pending:
            print(f"  {fp.name} ...", end=" ", flush=True)
            fp, r = _process(fp)
            if r:
                results.append(r)
                ok += 1
                print(f"OK [{r['_best_score']}pts, {r['_n_samples']}/{n} parsed]")
            else:
                fail += 1
                print("FAIL")
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {ok} extracted, {fail} failed.")
    print(f"Output: {out_file}  ({len(results)} total records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Legal metadata extraction via Groq {MODEL} with best-of-N expert sampling"
    )
    parser.add_argument("--folder",  type=Path, required=True,
                        help="Folder of documents to process")
    parser.add_argument("--output",  type=Path,
                        default=Path("data/expert_labels/results.json"),
                        help="Output JSON file (default: data/expert_labels/results.json)")
    parser.add_argument("--n",       type=int, default=N_SAMPLES,
                        help=f"Best-of-N samples per document (default: {N_SAMPLES})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel threads (default 1)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip docs already in the output file")
    parser.add_argument("--model",   default=MODEL,
                        help=f"Groq model override (default: {MODEL})")
    args = parser.parse_args()

    MODEL = args.model
    run(folder=args.folder, out_file=args.output,
        resume=args.resume, workers=args.workers, n=args.n)
