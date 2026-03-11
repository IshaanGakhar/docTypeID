"""
Cluster-aware ingestion pipeline.

Workflow:
    1. Load the cluster CSV — parse all pre-extracted metadata columns.
    2. Drop singleton and noise clusters.
    3. Run the extraction pipeline on every document (raw, no CSV merge yet).
    4. Compute cluster-level consensus from the RAW pipeline results only.
       (Computing it before CSV merge prevents CSV values from biasing
       what "the cluster thinks".)
    5. For every document, run three-way verification:
         pipeline result  vs  CSV pre-extracted values  vs  cluster consensus
       The winner is chosen by agreement:
         all three agree     → confidence 0.95
         pipeline + CSV      → confidence 0.90
         pipeline + consensus → confidence 0.85
         CSV + consensus (pipeline wrong/absent) → use CSV, confidence 0.78–0.80
         only one source     → use it, no boost
    6. Party names are verified against CSV all_entities and the CSV caption.
    7. Cluster-level metadata (anchor_uid, cluster_stage, …) is attached.

Expected CSV columns:
    cluster_id, cluster_stage, cluster_size, anchor_uid, cluster_pool_size,
    doc_origin, merged_from_s2_count, ground_truth_folder,
    majority_folder_in_cluster, is_misclassified,
    document_filename, document_path,
    num_raw_dockets, docket_primary_raw, docket_primary_normalized,
    docket_source, all_dockets_raw, all_dockets_normalized,
    num_westlaw_ids, all_westlaw_ids, num_lexis_ids, all_lexis_ids,
    num_uids_total, has_any_uid, num_entities, all_entities,
    doc_overlap_with_cluster, caption, court, location, ocr_applied
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

from pipeline.court_judge_extractor import _clean_court_name, _clean_location

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_CLUSTER_IDS: set[str] = {"-1", "-1.0", "noise", "none", "nan", "singleton", ""}

# Minimum Jaccard token-overlap to consider two string values as "matching"
_SIM_THRESHOLD = 0.35

# Legal stopwords — stripped before similarity comparison
_LEGAL_STOP = {
    "of", "the", "for", "in", "and", "or", "at", "to", "a", "an",
    "united", "states", "court", "district", "northern", "southern",
    "eastern", "western", "central", "middle",
}


# ---------------------------------------------------------------------------
# Per-document CSV record
# ---------------------------------------------------------------------------

@dataclass
class CsvDocRecord:
    """All fields parsed from a single CSV row."""
    file_path:    Path
    cluster_id:   str

    nuid:                    Optional[str] = None
    nuid_raw:                Optional[str] = None
    nuid_source:             Optional[str] = None
    all_dockets_normalized:  list[str]    = field(default_factory=list)
    all_dockets_raw:         list[str]    = field(default_factory=list)
    westlaw_ids:             list[str]    = field(default_factory=list)
    lexis_ids:               list[str]    = field(default_factory=list)

    court_name:     Optional[str] = None
    court_location: Optional[str] = None
    caption:        Optional[str] = None

    cluster_size:       int            = 1
    cluster_stage:      Optional[str]  = None
    anchor_uid:         Optional[str]  = None
    cluster_pool_size:  Optional[int]  = None
    majority_folder:    Optional[str]  = None
    ground_truth_folder: Optional[str] = None
    is_misclassified:   Optional[bool] = None

    doc_origin:            Optional[str]   = None
    merged_from_s2_count:  Optional[int]   = None
    doc_overlap_with_cluster: Optional[float] = None
    ocr_applied:           bool             = False
    all_entities:          list[str]        = field(default_factory=list)
    num_entities:          int              = 0


# ---------------------------------------------------------------------------
# CSV column detection helpers
# ---------------------------------------------------------------------------

def _col(header: list[str], *keywords: str) -> Optional[int]:
    for i, col in enumerate(header):
        col_l = col.strip().lower()
        if any(k in col_l for k in keywords):
            return i
    return None


def _val(row: list[str], idx: Optional[int], default: str = "") -> str:
    if idx is None or idx >= len(row):
        return default
    return row[idx].strip()


def _bool_val(raw: str) -> Optional[bool]:
    if raw.lower() in ("true", "1", "yes"):
        return True
    if raw.lower() in ("false", "0", "no"):
        return False
    return None


def _int_val(raw: str) -> Optional[int]:
    try:
        return int(float(raw)) if raw else None
    except ValueError:
        return None


def _float_val(raw: str) -> Optional[float]:
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _list_val(raw: str) -> list[str]:
    if not raw:
        return []
    sep = "|" if "|" in raw else ","
    return [x.strip() for x in raw.split(sep) if x.strip()]


def _detect_column_map(header: list[str]) -> dict[str, Optional[int]]:
    h = [c.strip().lower() for c in header]

    def exact(name: str) -> Optional[int]:
        try:
            return h.index(name)
        except ValueError:
            return None

    return {
        "document_path":   exact("document_path")   or _col(header, "document_path", "filepath", "file_path", "path", "file", "doc"),
        "cluster_id":      exact("cluster_id")       or _col(header, "cluster_id", "cluster", "label", "group"),
        "docket_primary_normalized": exact("docket_primary_normalized") or _col(header, "normalized", "nuid"),
        "docket_primary_raw":        exact("docket_primary_raw")        or _col(header, "docket_primary_raw"),
        "docket_source":             exact("docket_source")             or _col(header, "docket_source"),
        "all_dockets_normalized":    exact("all_dockets_normalized")    or _col(header, "all_dockets_normalized"),
        "all_dockets_raw":           exact("all_dockets_raw")           or _col(header, "all_dockets_raw"),
        "all_westlaw_ids":           exact("all_westlaw_ids")           or _col(header, "westlaw"),
        "all_lexis_ids":             exact("all_lexis_ids")             or _col(header, "lexis"),
        "court":           exact("court")            or _col(header, "court_name", "court"),
        "location":        exact("location")         or _col(header, "court_location", "location"),
        "caption":         exact("caption"),
        "cluster_size":         exact("cluster_size"),
        "cluster_stage":        exact("cluster_stage"),
        "anchor_uid":           exact("anchor_uid"),
        "cluster_pool_size":    exact("cluster_pool_size"),
        "majority_folder_in_cluster": exact("majority_folder_in_cluster") or _col(header, "majority_folder"),
        "ground_truth_folder":  exact("ground_truth_folder"),
        "is_misclassified":     exact("is_misclassified"),
        "doc_origin":                exact("doc_origin"),
        "merged_from_s2_count":      exact("merged_from_s2_count"),
        "doc_overlap_with_cluster":  exact("doc_overlap_with_cluster"),
        "ocr_applied":               exact("ocr_applied"),
        "all_entities":              exact("all_entities"),
        "num_entities":              exact("num_entities"),
    }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_cluster_csv(
    csv_path: Union[str, Path],
    base_folder: Union[str, Path, None] = None,
) -> dict[str, list[CsvDocRecord]]:
    csv_path    = Path(csv_path)
    base_folder = Path(base_folder) if base_folder else csv_path.parent

    sample  = csv_path.read_text(encoding="utf-8", errors="replace")[:8192]
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")

    raw_records: list[tuple] = []

    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, dialect)
        header = next(reader)
        col    = _detect_column_map(header)

        path_idx    = col["document_path"]
        cluster_idx = col["cluster_id"]

        if path_idx is None:
            raise ValueError(
                f"Cannot find file-path column in CSV header: {header}\n"
                "Expected a column named 'document_path', 'file_path', or 'path'."
            )
        if cluster_idx is None:
            raise ValueError(
                f"Cannot find cluster column in CSV header: {header}\n"
                "Expected a column named 'cluster_id' or 'cluster'."
            )

        for row in reader:
            if not row or len(row) <= max(path_idx, cluster_idx):
                continue
            fp_str = row[path_idx].strip()
            cid    = row[cluster_idx].strip()
            if fp_str:
                raw_records.append((fp_str, cid, header, row))

    cid_counts: Counter = Counter(cid for _, cid, _, _ in raw_records)

    pre_records: list[CsvDocRecord] = []
    dropped_noise     = 0
    dropped_singleton = 0

    for fp_str, cid, hdr, row in raw_records:
        cid_norm = cid.lower().strip()
        if cid_norm in NOISE_CLUSTER_IDS:
            dropped_noise += 1
            continue
        if cid_counts[cid] < 2:
            dropped_singleton += 1
            continue

        p = Path(fp_str)
        if not p.is_absolute():
            p = base_folder / p

        c = col

        def v(key: str) -> str:
            return _val(row, c.get(key))

        nuid_norm = v("docket_primary_normalized") or None
        nuid_raw  = v("docket_primary_raw") or None

        record = CsvDocRecord(
            file_path=p,
            cluster_id=cid,
            nuid=nuid_norm if nuid_norm and nuid_norm.lower() not in ("none", "nan", "") else None,
            nuid_raw=nuid_raw if nuid_raw and nuid_raw.lower() not in ("none", "nan", "") else None,
            nuid_source=v("docket_source") or None,
            all_dockets_normalized=_list_val(v("all_dockets_normalized")),
            all_dockets_raw=_list_val(v("all_dockets_raw")),
            westlaw_ids=_list_val(v("all_westlaw_ids")),
            lexis_ids=_list_val(v("all_lexis_ids")),
            court_name=(_clean_court_name(v("court")) or None) if v("court") else None,
            court_location=(_clean_location(v("location")) or None) if v("location") else None,
            caption=v("caption") or None,
            cluster_size=_int_val(v("cluster_size")) or cid_counts[cid],
            cluster_stage=v("cluster_stage") or None,
            anchor_uid=v("anchor_uid") or None,
            cluster_pool_size=_int_val(v("cluster_pool_size")),
            majority_folder=v("majority_folder_in_cluster") or None,
            ground_truth_folder=v("ground_truth_folder") or None,
            is_misclassified=_bool_val(v("is_misclassified")),
            doc_origin=v("doc_origin") or None,
            merged_from_s2_count=_int_val(v("merged_from_s2_count")),
            doc_overlap_with_cluster=_float_val(v("doc_overlap_with_cluster")),
            ocr_applied=_bool_val(v("ocr_applied")) or False,
            all_entities=_list_val(v("all_entities")),
            num_entities=_int_val(v("num_entities")) or 0,
        )
        pre_records.append(record)

    cluster_map: dict[str, list[CsvDocRecord]] = {}
    for rec in pre_records:
        cluster_map.setdefault(rec.cluster_id, []).append(rec)

    total_in  = len(raw_records)
    total_out = sum(len(v) for v in cluster_map.values())
    print(
        f"CSV loaded: {total_in} rows → "
        f"{len(cluster_map)} clusters, {total_out} documents kept "
        f"({dropped_noise} noise, {dropped_singleton} singletons dropped)",
        file=sys.stderr,
    )

    missing = [rec for recs in cluster_map.values() for rec in recs if not rec.file_path.exists()]
    if missing:
        print(
            f"\n  WARNING: {len(missing)} file(s) do not exist on disk "
            f"and will be skipped with skip_reason='file_not_found'.",
            file=sys.stderr,
        )
        for rec in missing[:10]:
            print(f"    MISSING: {rec.file_path}", file=sys.stderr)
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more.", file=sys.stderr)
        print(
            "  TIP: Pass --base-folder /your/docs/root/ if the CSV uses relative paths.\n",
            file=sys.stderr,
        )

    return cluster_map


# ---------------------------------------------------------------------------
# Cluster-level consensus — computed on RAW pipeline results only
# ---------------------------------------------------------------------------

def _clean_str(val: str) -> str:
    """Collapse all whitespace (including newlines) to a single space."""
    return " ".join(val.split()) if isinstance(val, str) else val


def _normalize_court(raw: str) -> str:
    """Canonical form for court names so surface variants group together.

    Strips common prefixes (IN THE, IN THE UNITED STATES), trailing OF,
    normalises whitespace and case.  E.g.:
        "IN THE DISTRICT COURT"           → "district court"
        "IN THE DISTRICT COURT OF"        → "district court"
        "District Court"                  → "district court"
        "UNITED STATES DISTRICT COURT"    → "united states district court"
        "IN THE UNITED STATES DISTRICT COURT" → "united states district court"
    """
    s = re.sub(r"\s+", " ", raw).strip().lower()
    s = re.sub(r"^in\s+the\s+", "", s)
    s = re.sub(r"\s+of\s*$", "", s)
    return s.strip()


def _normalize_nuid(raw: str) -> str:
    """Canonical form for docket/case numbers so formatting variants group together.

    Strips leading zeros from the year component, removes all separators
    (hyphens, spaces, periods), and lowercases.  E.g.:
        "18-cv-339231" → "18cv339231"
        "1:20-cv-05865-NRB" → "120cv05865nrb"
    """
    s = re.sub(r"[\-\s\.\:]", "", raw).lower()
    # Strip leading zeros in first numeric segment (year): "08cv" → "8cv"
    s = re.sub(r"^0+(\d)", r"\1", s)
    return s


def _majority(values: list, normalize_fn=None) -> object:
    candidates = [v for v in values if v is not None and v != "" and v != []]
    if not candidates:
        return None
    if all(isinstance(v, str) for v in candidates):
        normed = [_clean_str(v) for v in candidates]
        if normalize_fn:
            keys = [normalize_fn(v) for v in normed]
        else:
            keys = [v.strip().lower() for v in normed]
        counts: Counter = Counter(keys)
        winner_key = counts.most_common(1)[0][0]
        # Return the longest original form that maps to the winner key
        best = None
        for v, k in zip(normed, keys):
            if k == winner_key:
                if best is None or len(v) > len(best):
                    best = v
        return best
    return Counter(str(v) for v in candidates).most_common(1)[0][0]


_PARTY_DEDUP_THRESHOLD = 0.60   # Jaccard similarity above which two party strings
                                # are treated as the same entity


def _party_tokens(name: str) -> set[str]:
    """Significant tokens for party-name dedup (strip legal boilerplate)."""
    _BOILERPLATE = {
        "and", "or", "on", "behalf", "of", "all", "others", "similarly",
        "situated", "individually", "the", "a", "an", "et", "al",
    }
    return {w for w in re.findall(r"[a-z]+", name.lower()) if w not in _BOILERPLATE and len(w) > 1}


def _party_jaccard(a: str, b: str) -> float:
    ta, tb = _party_tokens(a), _party_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _union_parties(results: list[dict]) -> dict[str, list[str]]:
    from pipeline.party_extractor import _is_valid_party  # avoid circular at module level

    merged: dict[str, list[str]] = {
        "plaintiffs": [], "defendants": [], "petitioners": [], "respondents": []
    }
    for r in results:
        parties = r.get("parties") or {}
        for role in merged:
            for name in parties.get(role, []):
                clean = _clean_str(name)
                # Re-validate every candidate — bad extractions from one document in
                # the cluster must not be broadcast to all siblings via consensus.
                if not clean or not _is_valid_party(clean):
                    continue

                # Token-similarity dedup: find the closest existing entry
                best_idx, best_sim = -1, 0.0
                for idx, existing in enumerate(merged[role]):
                    sim = _party_jaccard(clean, existing)
                    if sim > best_sim:
                        best_sim, best_idx = sim, idx

                if best_sim >= _PARTY_DEDUP_THRESHOLD:
                    # Near-duplicate: keep whichever form is longer (more complete)
                    if len(clean) > len(merged[role][best_idx]):
                        merged[role][best_idx] = clean
                else:
                    merged[role].append(clean)
    return merged


def consolidate_cluster(results: list[dict]) -> dict:
    """
    Compute cluster-level consensus from raw pipeline results.

    document_types: only types present in >50% of cluster documents
    (prevents one outlier doc from adding its types to every sibling).
    All other scalar fields: simple majority vote.
    """
    non_skipped = [r for r in results if not r.get("skipped")]
    if not non_skipped:
        return {}

    n = len(non_skipped)

    # Document types: majority only (>50% of cluster docs)
    dt_counts: Counter = Counter()
    for r in non_skipped:
        for dt in set(r.get("document_types", [])):
            dt_counts[dt] += 1
    majority_doctypes = [dt for dt, cnt in dt_counts.items() if cnt / n > 0.5]

    def _vals(f: str) -> list:
        return [r.get(f) for r in non_skipped]

    return {
        "nuid":           _majority(_vals("nuid"), normalize_fn=_normalize_nuid),
        "court_name":     _majority(_vals("court_name"), normalize_fn=_normalize_court),
        "court_location": _majority(_vals("court_location")),
        "judge_name":     _majority(_vals("judge_name")),
        "document_types": majority_doctypes,
        "parties":        _union_parties(non_skipped),
        "n_docs":         n,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _token_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on significant word tokens (case-insensitive).
    Legal stopwords are stripped before comparison.
    """
    def tokens(s: str) -> set[str]:
        return {w for w in re.findall(r'\w+', s.lower())
                if w not in _LEGAL_STOP and len(w) > 1}
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _verify_scalar(
    pipeline_val: Optional[str],
    csv_val: Optional[str],
    consensus_val: Optional[str],
) -> tuple[Optional[str], Optional[float], str]:
    """
    Three-way vote for a single scalar string field.

    Returns (final_value, confidence_override_or_None, evidence_tag).

    Confidence overrides:
      pipeline + csv + consensus agree  → 0.95
      pipeline + csv agree              → 0.90
      pipeline + consensus agree        → 0.85
      csv + consensus agree (override)  → 0.78
      csv fills null (+ consensus agree)→ 0.80
      csv fills null (alone)            → 0.70
      consensus fills null              → 0.50
      no change                         → None (keep existing confidence)
    """
    def match(a: Optional[str], b: Optional[str]) -> bool:
        return bool(a and b and _token_similarity(a, b) >= _SIM_THRESHOLD)

    p_csv = match(pipeline_val, csv_val)
    p_con = match(pipeline_val, consensus_val)
    c_con = match(csv_val,      consensus_val)

    def _pick_most_specific(*vals: Optional[str]) -> Optional[str]:
        """Among agreeing values, prefer the longest (most detailed) form."""
        candidates = [v for v in vals if v]
        return max(candidates, key=len) if candidates else None

    if pipeline_val:
        if p_csv and p_con:
            best = _pick_most_specific(pipeline_val, csv_val, consensus_val)
            return best, 0.95, "pipeline+csv+consensus"
        elif p_csv:
            best = _pick_most_specific(pipeline_val, csv_val)
            return best, 0.90, "pipeline+csv"
        elif p_con:
            best = _pick_most_specific(pipeline_val, consensus_val)
            return best, 0.85, "pipeline+consensus"
        elif c_con:
            best = _pick_most_specific(csv_val, consensus_val)
            return best, 0.78, "csv+consensus_override"
        else:
            return pipeline_val, None, "pipeline_only"
    else:
        if csv_val and consensus_val and c_con:
            best = _pick_most_specific(csv_val, consensus_val)
            return best, 0.80, "csv+consensus_fill"
        elif csv_val:
            return csv_val, 0.70, "csv_fill"
        elif consensus_val:
            return consensus_val, 0.50, "consensus_fill"
        return None, None, "none"


def _entity_token_set(entities: list[str]) -> set[str]:
    """Build a flat token set from a list of entity strings."""
    stop = {"of", "the", "for", "in", "and", "or", "a", "an",
            "inc", "llc", "corp", "co", "ltd"}
    tokens: set[str] = set()
    for e in entities:
        tokens.update(w for w in re.findall(r'\w+', e.lower())
                      if w not in stop and len(w) > 2)
    return tokens


def _party_confidence_from_entities(name: str, entity_tokens: set[str]) -> float:
    """
    How well does an extracted party name overlap with the CSV all_entities pool?
    Returns a confidence float.
    """
    if not entity_tokens:
        return 0.70   # no entity info → neutral
    stop = {"of", "the", "for", "in", "and", "or", "a", "an"}
    name_tokens = {w for w in re.findall(r'\w+', name.lower()) if w not in stop and len(w) > 2}
    if not name_tokens:
        return 0.70
    overlap = len(name_tokens & entity_tokens) / len(name_tokens)
    if overlap >= 0.6:
        return 0.90   # strongly corroborated
    elif overlap >= 0.3:
        return 0.75   # partial match
    return 0.55       # low overlap → likely noise


def _parse_caption_parties(caption: Optional[str]) -> dict[str, list[str]]:
    """
    Parse 'PLAINTIFF v. DEFENDANT' from the CSV caption field.
    Returns {"plaintiffs": [...], "defendants": [...]}.
    """
    result: dict[str, list[str]] = {"plaintiffs": [], "defendants": []}
    if not caption:
        return result
    parts = re.split(r'\bv(?:s?\.?|ersus)\b', caption, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        left  = parts[0].strip().rstrip(",").strip()
        right = parts[1].strip().rstrip(",").strip()
        if left  and len(left)  < 150:
            result["plaintiffs"] = [left]
        if right and len(right) < 150:
            result["defendants"] = [right]
    return result


# ---------------------------------------------------------------------------
# Three-way verification + augmentation (replaces _merge_csv_metadata
# and _fill_from_consensus from the old design)
# ---------------------------------------------------------------------------

def _verify_and_augment(
    pipeline_result: dict,
    rec: CsvDocRecord,
    consensus: dict,
) -> dict:
    """
    Cross-validate pipeline output against CSV metadata and cluster consensus,
    then produce the final result for one document.

    Sources:
      pipeline  — what our regex/TF-IDF extractors found
      CSV       — pre-extracted values (court, location, caption, all_entities)
      consensus — cluster majority of pipeline results (computed before this call)

    Rules:
      NUID         → CSV is authoritative (already normalized + validated)
      court_name   → three-way vote
      court_location → three-way vote; CSV location + cluster consensus are reliable
      judge_name   → pipeline + consensus (no CSV judge column)
      filing_date  → pipeline primary; cluster consensus fills when pipeline finds nothing
      parties      → pipeline result, verified/scored against CSV all_entities;
                     CSV caption used to supplement if pipeline found nothing;
                     cluster consensus parties used to fill missing roles
      document_types → this doc's types + cluster majority types (>50% of cluster)
    """
    r = dict(pipeline_result)
    r["evidence"]   = dict(r.get("evidence", {}))
    r["confidence"] = dict(r.get("confidence", {}))

    _EV_BASE = {"page": 0, "char_start": 0, "char_end": 0}

    def _add_ev(f: str, source: str, span: str, rule_id: str) -> None:
        r["evidence"].setdefault(f, []).append(
            {"source": source, "span_text": str(span), "rule_id": rule_id, **_EV_BASE}
        )

    # ── 1. NUID — CSV authoritative (skip three-way vote) ────────────────────
    if rec.nuid:
        r["nuid"] = rec.nuid
        r["confidence"]["nuid"] = 0.97
        r["evidence"]["nuid"] = [{
            "source": "csv_metadata", "page": 0, "char_start": 0, "char_end": 0,
            "span_text": rec.nuid_raw or rec.nuid,
            "rule_id":  f"csv:docket_primary_normalized:{rec.nuid_source or 'unknown'}",
        }]

    # ── 2. Scalar fields — three-way vote ────────────────────────────────────
    _SCALAR_FIELDS = [
        # (output_field, pipeline_val, csv_val, consensus_val)
        ("court_name",     r.get("court_name"),     rec.court_name,     consensus.get("court_name")),
        ("court_location", r.get("court_location"), rec.court_location, consensus.get("court_location")),
        # judge: no CSV source, but cluster consensus is useful (same judge per case)
        ("judge_name",     r.get("judge_name"),     None,               consensus.get("judge_name")),
        # filing_date: each document keeps its own pipeline-extracted date.
        # Cluster consensus is NOT applied — docs in the same cluster often have
        # different filing/service dates and consensus overwrites them incorrectly.
    ]

    for field_name, p_val, c_val, k_val in _SCALAR_FIELDS:
        final_val, conf_override, ev_tag = _verify_scalar(p_val, c_val, k_val)

        if final_val != p_val:
            r[field_name] = final_val

        if conf_override is not None:
            r["confidence"][field_name] = conf_override

        if ev_tag not in ("pipeline_only", "none") and final_val:
            _add_ev(field_name, ev_tag, final_val, f"verify:{field_name}:{ev_tag}")

    # ── 3. Parties — entity-aware verification ────────────────────────────────
    entity_tokens = _entity_token_set(rec.all_entities)
    parties = {k: list(v) for k, v in (r.get("parties") or {}).items()}
    party_conf: dict[str, float] = {}

    # Score each extracted party name against the CSV entity pool
    for role, names in parties.items():
        for name in names:
            conf = _party_confidence_from_entities(name, entity_tokens)
            party_conf[f"{role}:{name}"] = conf

    # If pipeline found no parties at all, try parsing the CSV caption
    has_any = any(parties.get(role) for role in ("plaintiffs", "defendants",
                                                   "petitioners", "respondents"))
    if not has_any and rec.caption:
        caption_parties = _parse_caption_parties(rec.caption)
        for role, names in caption_parties.items():
            for name in names:
                parties.setdefault(role, []).append(name)
                party_conf[f"{role}:{name}"] = 0.75  # caption is reliable
                _add_ev("parties", "csv_caption",
                        name, f"csv:caption:{role}")

    # Supplement missing roles from cluster consensus — only when the pipeline
    # found nothing for that role AND the CSV caption fallback above also failed.
    # This covers procedural documents (service, notices) that lack a caption.
    consensus_parties = consensus.get("parties", {})
    for role in ("plaintiffs", "defendants", "petitioners", "respondents"):
        if not parties.get(role) and consensus_parties.get(role):
            parties[role] = consensus_parties[role]
            for name in consensus_parties[role]:
                party_conf[f"{role}:{name}"] = 0.55
                _add_ev("parties", "cluster_consensus",
                        name, f"cluster_consensus:parties.{role}")

    r["parties"] = parties
    # Store per-name entity-overlap scores separately so confidence["parties"]
    # remains a scalar float (as the schema expects).
    if party_conf:
        r["party_entity_scores"] = party_conf

    # ── 4. Document types — add cluster majority types ────────────────────────
    existing_types: set[str] = set(r.get("document_types") or [])
    for dt in consensus.get("document_types", []):
        if dt not in existing_types:
            # Only add if >50% of cluster docs have it (enforced in consolidate_cluster)
            existing_types.add(dt)
            r.setdefault("document_types", []).append(dt)
            _add_ev("document_types", "cluster_consensus",
                    dt, "cluster_consensus:document_types:majority")

    # ── 5. CSV passthrough fields (always attached for auditability) ──────────
    r["csv_caption"]      = rec.caption
    r["csv_all_dockets"]  = rec.all_dockets_normalized
    r["csv_westlaw_ids"]  = rec.westlaw_ids
    r["csv_lexis_ids"]    = rec.lexis_ids
    r["csv_all_entities"] = rec.all_entities
    r["csv_doc_origin"]   = rec.doc_origin
    r["csv_ocr_applied"]  = rec.ocr_applied

    return r


# ---------------------------------------------------------------------------
# Worker shim — runs pipeline only, no CSV merge (consensus not known yet)
# ---------------------------------------------------------------------------

def _process_file_cluster(args: tuple) -> dict:
    """
    Run the extraction pipeline for one document.
    CSV metadata is NOT merged here — that happens later in _verify_and_augment
    after cluster consensus has been computed from clean pipeline results.
    """
    from pipeline.run_pipeline import run_pipeline, _skip_result

    rec_dict, min_chars, threshold = args
    file_path  = rec_dict["file_path"]
    cluster_id = rec_dict["cluster_id"]

    try:
        result = run_pipeline(
            file_path,
            min_chars=min_chars,
            doctype_threshold=threshold,
            skip_nuid=(rec_dict.get("nuid") is not None),
        )
    except Exception as exc:
        result = _skip_result(str(file_path), f"error:{exc}")

    result["cluster_id"] = cluster_id
    return result


# ---------------------------------------------------------------------------
# Main cluster pipeline
# ---------------------------------------------------------------------------

def run_pipeline_clusters(
    cluster_map: dict[str, list[CsvDocRecord]],
    workers: int = 1,
    min_chars: int = 150,
    doctype_threshold: float = 0.5,
    output_path: Union[str, Path, None] = None,
    progress: bool = True,
) -> list[dict]:
    """
    Process all documents and return verified, augmented result dicts.

    Pipeline:
      1. Run extraction on every document (parallel if workers > 1).
      2. Compute cluster consensus from raw pipeline results.
      3. Run _verify_and_augment (three-way vote + entity verification).
      4. Attach cluster-level metadata fields.
    """
    import json as _json

    total_docs     = sum(len(v) for v in cluster_map.values())
    total_clusters = len(cluster_map)

    if progress:
        print(f"Processing {total_docs} documents across {total_clusters} clusters",
              file=sys.stderr)

    # Serialize CsvDocRecord → dict for multiprocessing pickling
    tasks: list[tuple] = []
    task_recs: list[CsvDocRecord] = []
    for recs in cluster_map.values():
        for rec in recs:
            rec_dict = asdict(rec)
            rec_dict["file_path"] = str(rec.file_path)
            tasks.append((rec_dict, min_chars, doctype_threshold))
            task_recs.append(rec)

    # Step 1 — Run pipeline (raw, no CSV merge)
    raw_results: list[dict] = [{}] * len(tasks)

    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(_process_file_cluster, t): i
                             for i, t in enumerate(tasks)}
            iterator = as_completed(future_to_idx)
            if progress and _has_tqdm:
                iterator = tqdm(iterator, total=total_docs,
                                desc="Extracting", unit="doc", file=sys.stderr)
            for future in iterator:
                idx = future_to_idx[future]
                raw_results[idx] = future.result()
    else:
        iterator = enumerate(tasks)
        if progress and _has_tqdm:
            iterator = tqdm(iterator, total=total_docs,
                            desc="Extracting", unit="doc", file=sys.stderr)
        for i, task in iterator:
            raw_results[i] = _process_file_cluster(task)
            if progress and _has_tqdm:
                r = raw_results[i]
                status = "SKIP" if r.get("skipped") else "OK"
                iterator.set_postfix_str(
                    f"[{status}] cluster {r.get('cluster_id', '?')} | "
                    f"{Path(str(task_recs[i].file_path)).name[-40:]}",
                    refresh=True,
                )
            elif progress:
                r      = raw_results[i]
                status = "SKIP" if r.get("skipped") else "OK"
                print(
                    f"  [{i + 1}/{total_docs}] [{status}] "
                    f"[cluster {r.get('cluster_id', '?')}] "
                    f"{Path(str(task_recs[i].file_path)).name}",
                    file=sys.stderr,
                )

    # Group by cluster
    by_cluster: dict[str, list[dict]]          = {}
    rec_by_cluster: dict[str, list[CsvDocRecord]] = {}
    for i, r in enumerate(raw_results):
        cid = r.get("cluster_id", "unknown")
        by_cluster.setdefault(cid, []).append(r)
        rec_by_cluster.setdefault(cid, []).append(task_recs[i])

    final_results: list[dict] = []

    for cid, cluster_results in by_cluster.items():
        # Step 2 — Consensus from raw pipeline results only
        consensus = consolidate_cluster(cluster_results)

        cluster_recs = rec_by_cluster[cid]
        first_rec    = cluster_recs[0]

        # NUID consensus: pipeline skips NUID when CSV supplies one, so
        # consensus.nuid is typically null. Populate it from CSV dockets.
        if not consensus.get("nuid"):
            csv_nuids = [rec.nuid for rec in cluster_recs if rec.nuid]
            if csv_nuids:
                consensus["nuid"] = _majority(csv_nuids, normalize_fn=_normalize_nuid)

        for r, rec in zip(cluster_results, cluster_recs):
            # Step 3 — Three-way verification + augmentation
            if r.get("skipped"):
                # Still attach CSV passthrough and cluster metadata to skipped docs
                augmented = dict(r)
                augmented["csv_caption"]      = rec.caption
                augmented["csv_all_dockets"]  = rec.all_dockets_normalized
                augmented["csv_westlaw_ids"]  = rec.westlaw_ids
                augmented["csv_lexis_ids"]    = rec.lexis_ids
                augmented["csv_all_entities"] = rec.all_entities
                augmented["csv_doc_origin"]   = rec.doc_origin
                augmented["csv_ocr_applied"]  = rec.ocr_applied
            else:
                augmented = _verify_and_augment(r, rec, consensus)

            # Step 4 — Cluster-level metadata
            augmented["cluster_id"]          = cid
            augmented["cluster_size"]        = len(cluster_results)
            augmented["cluster_stage"]       = first_rec.cluster_stage
            augmented["anchor_uid"]          = first_rec.anchor_uid
            augmented["cluster_pool_size"]   = first_rec.cluster_pool_size
            augmented["majority_folder"]     = first_rec.majority_folder
            augmented["ground_truth_folder"] = first_rec.ground_truth_folder
            augmented["is_misclassified"]    = first_rec.is_misclassified
            augmented["cluster_consensus"]   = {
                k: v for k, v in consensus.items() if k != "n_docs"
            }
            final_results.append(augmented)

    if progress:
        skipped = sum(1 for r in final_results if r.get("skipped"))
        print(
            f"\nDone. {total_docs - skipped} processed, {skipped} skipped "
            f"across {total_clusters} clusters.",
            file=sys.stderr,
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            _json.dump(final_results, fh, indent=2, ensure_ascii=False)
        if progress:
            print(f"Output → {output_path}", file=sys.stderr)

    return final_results
