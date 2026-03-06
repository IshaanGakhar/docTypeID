"""
Pipeline orchestrator.

Two ingestion modes:

  CLUSTER MODE (primary)  — recommended
      Pass a CSV mapping file_path → cluster_id.
      Singletons and noise clusters are dropped automatically.
      Cluster-level consensus metadata fills nulls in individual documents.

      python -m pipeline.run_pipeline --cluster-csv clusters.csv --output results.json
      python -m pipeline.run_pipeline --cluster-csv clusters.csv --folder /docs/ --output results.json

  FOLDER MODE  — legacy / exploratory
      Scan a folder for all supported documents.

      python -m pipeline.run_pipeline --folder /path/to/docs/ --output results.json
      python -m pipeline.run_pipeline --folder /path/to/docs/ --workers 4

Single-document run_pipeline() is still importable for programmatic use.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Union

from pipeline.config import MIN_CHARS, DOCTYPE_THRESHOLD
from pipeline.pdf_loader import load_document, LoadedDocument, SUPPORTED_EXTENSIONS
from pipeline.preprocess import preprocess, DocumentZones, lines_to_text
from pipeline.title_extractor import extract_title, TitleResult
from pipeline.doc_type_classifier import classify_document_type
from pipeline.uid_extractor import extract_nuid
from pipeline.court_judge_extractor import extract_court_and_judge
from pipeline.date_extractor import extract_filing_date
from pipeline.party_extractor import extract_parties
from pipeline.clause_extractor import extract_clauses
from pipeline.crf_ner import extract_entities_crf


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

def _clean_str(val: object) -> object:
    """
    Collapse all whitespace (including newlines) in a string to a single space.
    Non-string values pass through unchanged.
    """
    if not isinstance(val, str):
        return val
    return " ".join(val.split())


def _deep_clean(obj: object) -> object:
    """Recursively replace every '\\n' (and all whitespace runs) in every string
    in the result tree with a single space.  Handles dicts, lists, and scalars."""
    if isinstance(obj, dict):
        return {k: _deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clean(item) for item in obj]
    return _clean_str(obj)


# ---------------------------------------------------------------------------
# Evidence serialization
# ---------------------------------------------------------------------------

def _ev_to_dict(ev_obj) -> dict:
    return {
        "source":     getattr(ev_obj, "source", ""),
        "page":       getattr(ev_obj, "page", 1),
        "span_text":  getattr(ev_obj, "span_text", ""),
        "char_start": getattr(ev_obj, "char_start", 0),
        "char_end":   getattr(ev_obj, "char_end", 0),
        "rule_id":    getattr(ev_obj, "rule_id", ""),
    }


# ---------------------------------------------------------------------------
# Skip result
# ---------------------------------------------------------------------------

def _skip_result(file_path: str, reason: str) -> dict:
    return {
        "skipped":    True,
        "skip_reason": reason,
        "file_path":  file_path,
        "title":      None,
        "document_types": [],
        "nuid":       None,
        "court_name": None,
        "court_location": None,
        "judge_name": None,
        "filing_date": None,
        "parties": {
            "plaintiffs":  [],
            "defendants":  [],
            "petitioners": [],
            "respondents": [],
        },
        "clauses":    [],
        "evidence":   {},
        "confidence": {},
    }


# ---------------------------------------------------------------------------
# Single-document pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: Union[str, Path],
    min_chars: int = MIN_CHARS,
    doctype_threshold: float = DOCTYPE_THRESHOLD,
    skip_nuid: bool = False,
) -> dict:
    """
    Run the full extraction pipeline on a single document or raw text string.

    Parameters
    ----------
    source            : path to a PDF / TXT / DOCX / DOC file, or a raw text string
    min_chars         : minimum characters threshold for skipping
    doctype_threshold : probability threshold for multi-label doc type

    Returns
    -------
    dict matching the output JSON schema (always JSON-serializable)
    """
    # Step 1 — Load
    loaded: LoadedDocument = load_document(
        source, min_chars=min_chars, extract_first_page_blocks=True
    )
    file_path = loaded.pdf_path

    if loaded.skipped:
        return _skip_result(file_path, loaded.skip_reason)

    full_text = loaded.full_text
    pages     = loaded.pages

    # Step 2 — Preprocess
    zones: DocumentZones = preprocess(loaded)

    # Step 3 — Title
    title_result: TitleResult = extract_title(zones)
    title = title_result.title

    # Step 4 — Document type
    # Restrict to first page only: the document type is declared in the heading,
    # not scattered through the body. Using full_text causes near-universal
    # matching of every type (e.g. "Order" appears in every filing).
    first_page_text = lines_to_text(zones.first_page_zone)
    doctype_result = classify_document_type(
        text=first_page_text,
        title=title or "",
        threshold=doctype_threshold,
    )

    # Step 5 — NUID
    # When the CSV already supplies a validated docket number, skip the regex
    # extractor entirely — the CSV value will be merged in by the cluster pipeline.
    if skip_nuid:
        from pipeline.uid_extractor import UIDResult
        uid_result = UIDResult(nuid=None, raw=None, source_tier=None,
                               confidence=0.0, evidence=[])
    else:
        uid_result = extract_nuid(full_text, pages=pages if pages else None)

    # Step 6 — CRF NER (no-op if model missing)
    caption_text = lines_to_text(zones.caption_zone)
    crf_result   = extract_entities_crf(caption_text, page_num=1)

    # Step 7 — Court + judge
    court_result = extract_court_and_judge(zones, crf_result=crf_result)

    # Step 8 — Date
    date_result = extract_filing_date(zones)

    # Step 9 — Parties
    party_result = extract_parties(zones)

    # Step 10 — Clauses
    clause_result = extract_clauses(zones)

    # Assemble evidence
    evidence: dict[str, list[dict]] = {}
    if title_result.evidence:
        evidence["title"]          = [_ev_to_dict(e) for e in title_result.evidence]
    if doctype_result.evidence:
        evidence["document_types"] = [_ev_to_dict(e) for e in doctype_result.evidence]
    if uid_result.evidence:
        evidence["nuid"]           = [_ev_to_dict(e) for e in uid_result.evidence]
    if court_result.court_evidence:
        evidence["court_name"]     = [_ev_to_dict(e) for e in court_result.court_evidence]
    if court_result.location_evidence:
        evidence["court_location"] = [_ev_to_dict(e) for e in court_result.location_evidence]
    if court_result.judge_evidence:
        evidence["judge_name"]     = [_ev_to_dict(e) for e in court_result.judge_evidence]
    if date_result.evidence:
        evidence["filing_date"]    = [_ev_to_dict(e) for e in date_result.evidence]
    if party_result.evidence:
        evidence["parties"]        = [_ev_to_dict(e) for e in party_result.evidence]
    if clause_result.evidence:
        evidence["clauses"]        = [_ev_to_dict(e) for e in clause_result.evidence]

    confidence: dict[str, float] = {
        "title":          title_result.confidence,
        "document_types": doctype_result.confidence,
        "nuid":           uid_result.confidence,
        "court_name":     court_result.court_confidence,
        "court_location": court_result.location_confidence,
        "judge_name":     court_result.judge_confidence,
        "filing_date":    date_result.confidence,
        "parties":        party_result.confidence,
    }

    clauses_out = [
        {
            "clause_type": c.clause_type,
            "heading":     c.heading,
            "text":        c.text,
            "page_start":  c.page_start,
            "page_end":    c.page_end,
        }
        for c in clause_result.clauses
    ]

    # Normalize parties: collapse newlines in each party name
    clean_parties: dict[str, list[str]] = {
        role: [str(_clean_str(n)) for n in names]
        for role, names in party_result.parties.items()
    }

    return {
        "skipped":        False,
        "file_path":      file_path,
        "title":          _clean_str(title),
        "document_types": doctype_result.document_types,
        "nuid":           _clean_str(uid_result.nuid),
        "court_name":     _clean_str(court_result.court_name),
        "court_location": _clean_str(court_result.court_location),
        "judge_name":     _clean_str(court_result.judge_name),
        "filing_date":    _clean_str(date_result.filing_date),
        "parties":        clean_parties,
        "clauses":        clauses_out,
        "evidence":       evidence,
        "confidence":     confidence,
    }


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def discover_files(folder: Path, recursive: bool = True) -> list[Path]:
    """
    Return all supported document files in *folder*.

    Parameters
    ----------
    folder    : directory to scan
    recursive : if True, scan all subdirectories; otherwise only the top level
    """
    glob = "**/*" if recursive else "*"
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"{glob}{ext}"))
    # Stable, deterministic order
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Worker shim (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _process_file(args: tuple) -> dict:
    """Unpacks (file_path, min_chars, threshold) and calls run_pipeline."""
    file_path, min_chars, threshold = args
    try:
        return run_pipeline(file_path, min_chars=min_chars, doctype_threshold=threshold)
    except Exception as exc:
        return _skip_result(str(file_path), f"error:{exc}")


# ---------------------------------------------------------------------------
# Folder pipeline
# ---------------------------------------------------------------------------

def run_pipeline_dir(
    folder: Union[str, Path],
    recursive: bool = True,
    workers: int = 1,
    min_chars: int = MIN_CHARS,
    doctype_threshold: float = DOCTYPE_THRESHOLD,
    output_path: Union[str, Path, None] = None,
    progress: bool = True,
) -> list[dict]:
    """
    Process every supported document in *folder* and return a list of result dicts.

    Parameters
    ----------
    folder            : directory containing documents
    recursive         : recurse into subdirectories (default True)
    workers           : parallel worker processes (default 1 = serial)
    min_chars         : per-document skip threshold
    doctype_threshold : document-type probability threshold
    output_path       : if given, write JSON array to this file
    progress          : print progress to stderr

    Returns
    -------
    list of result dicts (one per discovered file)
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files = discover_files(folder, recursive=recursive)
    total = len(files)

    if total == 0:
        if progress:
            print(f"No supported documents found in '{folder}'", file=sys.stderr)
        return []

    if progress:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(
            f"Found {total} document(s) in '{folder}' "
            f"(extensions: {exts}, recursive={recursive})",
            file=sys.stderr,
        )

    task_args = [(str(f), min_chars, doctype_threshold) for f in files]
    results: list[dict] = [{}] * total

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_process_file, arg): i
                for i, arg in enumerate(task_args)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                done += 1
                if progress:
                    print(
                        f"  [{done}/{total}] {files[idx].name}",
                        file=sys.stderr,
                    )
    else:
        for i, arg in enumerate(task_args):
            results[i] = _process_file(arg)
            if progress:
                status = "SKIP" if results[i].get("skipped") else "OK"
                print(
                    f"  [{i + 1}/{total}] [{status}] {files[i].name}",
                    file=sys.stderr,
                )

    results = [_deep_clean(r) for r in results]  # type: ignore[assignment]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        if progress:
            skipped = sum(1 for r in results if r.get("skipped"))
            print(
                f"\nDone. {total - skipped} processed, {skipped} skipped. "
                f"Output → {output_path}",
                file=sys.stderr,
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Legal document extraction pipeline.\n\n"
            "CLUSTER MODE (recommended):\n"
            "  python -m pipeline.run_pipeline --cluster-csv clusters.csv --output results.json\n\n"
            "FOLDER MODE:\n"
            "  python -m pipeline.run_pipeline --folder /path/to/docs/ --output results.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Source (mutually exclusive) ─────────────────────────────────────────
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--cluster-csv", metavar="CSV",
        help=(
            "CSV file mapping file_path → cluster_id. "
            "Singletons and noise clusters (-1) are dropped automatically."
        ),
    )
    source_group.add_argument(
        "--folder", metavar="DIR",
        help="Folder containing documents (.pdf/.txt/.docx/.doc).",
    )

    # ── Common options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write JSON array to this file (default: print JSONL to stdout)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1,
        help="Parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=MIN_CHARS,
        help=f"Skip documents with fewer than N characters (default: {MIN_CHARS})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DOCTYPE_THRESHOLD,
        help=f"Document-type probability threshold (default: {DOCTYPE_THRESHOLD})",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )

    # ── Folder-mode options ─────────────────────────────────────────────────
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false", default=True,
        help="(folder mode) Do not recurse into subdirectories",
    )

    # ── Cluster-mode options ────────────────────────────────────────────────
    parser.add_argument(
        "--base-folder", metavar="DIR", default=None,
        help=(
            "(cluster mode) Base folder for resolving relative paths in the CSV. "
            "Defaults to the CSV file's own directory."
        ),
    )

    args = parser.parse_args()

    # ── Cluster mode ────────────────────────────────────────────────────────
    if args.cluster_csv:
        from pipeline.cluster_ingestion import load_cluster_csv, run_pipeline_clusters

        csv_path = Path(args.cluster_csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        cluster_map = load_cluster_csv(
            csv_path,
            base_folder=args.base_folder,
        )
        if not cluster_map:
            print("No clusters remaining after dropping singletons.", file=sys.stderr)
            sys.exit(0)

        results = run_pipeline_clusters(
            cluster_map=cluster_map,
            workers=args.workers,
            min_chars=args.min_chars,
            doctype_threshold=args.threshold,
            output_path=args.output,
            progress=not args.quiet,
        )

    # ── Folder mode ─────────────────────────────────────────────────────────
    else:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"ERROR: path does not exist: {folder}", file=sys.stderr)
            sys.exit(1)
        if not folder.is_dir():
            print(f"ERROR: '{folder}' is a file, not a folder.", file=sys.stderr)
            sys.exit(1)

        results = run_pipeline_dir(
            folder=folder,
            recursive=args.recursive,
            workers=args.workers,
            min_chars=args.min_chars,
            doctype_threshold=args.threshold,
            output_path=args.output,
            progress=not args.quiet,
        )

    # Stream JSONL to stdout if no --output
    if not args.output:
        for result in results:
            print(json.dumps(_deep_clean(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
