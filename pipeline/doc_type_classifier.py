"""
Multi-label document type classifier.

Primary: TF-IDF (word + char) + OneVsRest (LogisticRegression | LinearSVC | ComplementNB)
Fallback: Rule-based regex detection when model files are missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import hstack

from pipeline.config import (
    DOCTYPE_THRESHOLD,
    DOCTYPE_RULES,
    CLASSIFIER_BACKEND,
    MODEL_DOCTYPE_PATH,
    LABEL_BINARIZER_PATH,
    VECTORIZER_WORD_PATH,
    VECTORIZER_CHAR_PATH,
    CITATION_BOILERPLATE_RE_STR,
)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class DocTypeEvidence:
    source: str       # "model" | "rule"
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class DocTypeResult:
    document_types: list[str]
    probabilities: dict[str, float]
    confidence: float
    evidence: list[DocTypeEvidence]
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Binary regex features (used both for fallback and as model features)
# ---------------------------------------------------------------------------

_BINARY_REGEX_FEATURES: dict[str, re.Pattern] = {
    "has_rule12b6":      re.compile(r"Rule\s+12\s*\(\s*b\s*\)\s*\(\s*6\s*\)", re.IGNORECASE),
    "has_notice_of":     re.compile(r"\bNotice\s+of\b", re.IGNORECASE),
    "has_motion_to":     re.compile(r"\bMotion\s+to\b", re.IGNORECASE),
    "has_in_support":    re.compile(r"\bin\s+support\b", re.IGNORECASE),
    "has_comes_now":     re.compile(r"\bComes\s+Now\b", re.IGNORECASE),
    "has_whereas":       re.compile(r"\bWHEREAS\b"),
    "has_ordered":       re.compile(r"\bIT\s+IS\s+(?:HEREBY\s+)?ORDERED\b"),
    "has_verified":      re.compile(r"\bVERIFIED\b"),
    "has_under_oath":    re.compile(r"\bunder\s+(?:penalty\s+of\s+)?perjury\b", re.IGNORECASE),
    "has_stipulated":    re.compile(r"\bSTIPULATED\b"),
    "has_judgment":      re.compile(r"\bJUDGMENT\b"),
    "has_default":       re.compile(r"\bDEFAULT\b"),
    "has_subpoena":      re.compile(r"\bSUBPOENA\b"),
    "has_transcript":    re.compile(r"\bTRANSCRIPT\b"),
    "has_proceedings":   re.compile(r"\bPROCEEDINGS\b"),
}


def compute_binary_features(text: str) -> np.ndarray:
    """Return a 1-D numpy array of binary flags for each regex feature."""
    return np.array(
        [1.0 if pat.search(text) else 0.0 for pat in _BINARY_REGEX_FEATURES.values()],
        dtype=np.float32,
    ).reshape(1, -1)


# ---------------------------------------------------------------------------
# Compound title splitting
# ---------------------------------------------------------------------------

# Split compound titles on " AND ", " & ", " OR " to check each sub-phrase
_TITLE_SPLIT_RE = re.compile(r"\s+(?:AND|&|OR)\s+", re.IGNORECASE)


def _title_segments(title: str) -> list[str]:
    """
    Split a compound title into sub-phrases for per-segment type matching.

    "MOTION TO DISMISS AND MEMORANDUM IN SUPPORT" →
        ["MOTION TO DISMISS", "MEMORANDUM IN SUPPORT"]

    The original full title is always included as the first entry so that
    patterns spanning the conjunction are not missed.
    """
    if not title:
        return []
    parts = [title] + [p.strip() for p in _TITLE_SPLIT_RE.split(title) if p.strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        key = p.upper()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_GENERIC_LABELS = {"Motion", "Order", "Complaint", "Brief", "Notice",
                   "Judgment", "Petition", "Answer", "Memorandum"}

# Leaf types whose keywords are specific enough that body-text matches are safe
_LEAF_TYPES = {k for k in (
    "Motion to Dismiss", "Motion to Stay", "Motion for Summary Judgment",
    "Motion to Compel", "Motion in Limine", "Motion to Strike",
    "Motion to Transfer", "Motion to Remand", "Motion for Class Certification",
    "Motion for Injunction", "Motion for Sanctions", "Motion for Reconsideration",
    "Motion for Leave to Amend", "Motion for Extension of Time",
    "Motion for Default Judgment", "Motion to Quash", "Motion to Intervene",
    "Motion for Judgment on the Pleadings", "Motion for Protective Order",
    "Temporary Restraining Order", "Preliminary Injunction", "Scheduling Order",
    "Amended Complaint", "Class Action Complaint",
    "Opening Brief", "Reply Brief", "Opposition Brief", "Appellate Brief",
    "Notice of Appeal", "Notice of Removal", "Notice of Motion",
    "Proposed Order", "Counterclaim", "Cross-Claim", "Third-Party Complaint",
    "Proof of Service", "Certificate of Service", "Cover Letter",
    "Demand Letter", "Letter Brief", "Letter Motion", "Letter to Court",
    "Civil Cover Sheet", "Letter", "Citation",
    "Pro Hac Vice Motion", "Response", "Entry of Appearance",
    "Corporate Disclosure Statement", "Order of Transfer",
    "Exhibit", "Retainer Agreement", "Notice of Voluntary Dismissal",
)}


def _rule_based_classify(text: str, title: str = "") -> DocTypeResult:
    """
    Classify by regex rules when model files are absent.

    Runs rules against:
      1. Title zone / title segments (high confidence)
      2. Body text for *leaf* (multi-word) types — safe, unlikely to false-positive
      3. Generic single-word labels (Order, Complaint, …) are only matched
         in the title zone (first ~500 chars) to prevent ubiquitous body
         references from triggering false positives.
    """
    body_upper = text[:3000].upper()
    title_zone_upper = text[:500].upper()
    matched: list[str] = []
    evidence: list[DocTypeEvidence] = []
    probs: dict[str, float] = {}

    title_segments = _title_segments(title)

    search_targets: list[tuple[str, str]] = []

    if title_segments:
        search_targets.append((title_segments[0].upper() + "\n" + title_zone_upper, "title_zone"))
        for seg in title_segments[1:]:
            search_targets.append((seg.upper(), f"title_seg:{seg[:30]}"))
    else:
        search_targets.append((title_zone_upper, "title_zone"))

    # Body target is only searched for leaf (multi-word) types
    search_targets.append((body_upper, "body"))

    matched_families: set[str] = set()

    for search_text, src_label in search_targets:
        for doc_type, patterns in DOCTYPE_RULES.items():
            if doc_type in matched:
                continue

            if doc_type in _GENERIC_LABELS and doc_type in matched_families:
                continue

            # Generic single-word labels must only match in title zone / title segments
            if doc_type in _GENERIC_LABELS and src_label == "body":
                continue

            for pat_str in patterns:
                m = re.search(pat_str, search_text)
                if m:
                    matched.append(doc_type)
                    conf = 0.90 if src_label.startswith("title_seg") else 0.85
                    probs[doc_type] = conf
                    evidence.append(DocTypeEvidence(
                        source="rule",
                        page=1,
                        span_text=m.group(0),
                        char_start=m.start(),
                        char_end=m.end(),
                        rule_id=f"doctype_rule:{doc_type}:{src_label}",
                    ))
                    for generic in _GENERIC_LABELS:
                        if doc_type.startswith(generic):
                            matched_families.add(generic)
                    break

    # Citation boilerplate override: when the first page is a Texas-style
    # citation/service form, suppress Petition/Answer/Judgment that appear
    # only because the form boilerplate mentions them as instructions.
    _CITATION_BP = re.compile(CITATION_BOILERPLATE_RE_STR, re.IGNORECASE)
    if _CITATION_BP.search(body_upper):
        _BOILERPLATE_TYPES = {"Petition", "Answer", "Judgment"}
        matched = [dt for dt in matched if dt not in _BOILERPLATE_TYPES]
        for dt in _BOILERPLATE_TYPES:
            probs.pop(dt, None)
        evidence = [
            e for e in evidence
            if not (":" in e.rule_id
                    and len(e.rule_id.split(":")) > 1
                    and e.rule_id.split(":")[1] in _BOILERPLATE_TYPES)
        ]
        if "Citation" not in matched:
            matched.insert(0, "Citation")
            probs["Citation"] = 0.92
            evidence.append(DocTypeEvidence(
                source="rule", page=1,
                span_text=_CITATION_BP.search(body_upper).group(0),
                char_start=0, char_end=0,
                rule_id="doctype_rule:Citation:boilerplate_override",
            ))

    matched, probs, evidence = _ensure_letter_base_tag(matched, probs, evidence)

    confidence = max(probs.values()) if probs else 0.0
    return DocTypeResult(
        document_types=matched,
        probabilities=probs,
        confidence=confidence,
        evidence=evidence,
        fallback_used=True,
    )


# ---------------------------------------------------------------------------
# Letter base-tag enforcement
# ---------------------------------------------------------------------------

_LETTER_SUBTYPES = {
    "Cover Letter", "Demand Letter", "Letter Brief",
    "Letter Motion", "Letter to Court",
}


def _ensure_letter_base_tag(
    matched: list[str],
    probs: dict[str, float],
    evidence: list[DocTypeEvidence],
) -> tuple[list[str], dict[str, float], list[DocTypeEvidence]]:
    """If any letter subtype is present, ensure 'Letter' base tag is too."""
    has_subtype = any(dt in _LETTER_SUBTYPES for dt in matched)
    if has_subtype and "Letter" not in matched:
        matched.append("Letter")
        probs["Letter"] = max(
            (probs.get(dt, 0.0) for dt in _LETTER_SUBTYPES if dt in probs),
            default=0.85,
        )
        evidence.append(DocTypeEvidence(
            source="rule", page=1, span_text="(inferred from letter subtype)",
            char_start=0, char_end=0,
            rule_id="doctype_rule:Letter:subtype_inference",
        ))
    return matched, probs, evidence


# ---------------------------------------------------------------------------
# Model-based inference
# ---------------------------------------------------------------------------

def _load_model():
    """Load vectorizers, classifier, and label binarizer. Returns None if missing."""
    try:
        import joblib
        clf   = joblib.load(MODEL_DOCTYPE_PATH)
        lb    = joblib.load(LABEL_BINARIZER_PATH)
        vec_w = joblib.load(VECTORIZER_WORD_PATH)
        vec_c = joblib.load(VECTORIZER_CHAR_PATH)
        return clf, lb, vec_w, vec_c
    except Exception:
        return None


def _model_classify(
    text: str,
    title: str,
    threshold: float,
) -> DocTypeResult:
    """Run model-based classification. Falls back to rules on any error."""
    loaded = _load_model()
    if loaded is None:
        return _rule_based_classify(text, title)

    clf, lb, vec_w, vec_c = loaded

    # For the model, use the full title + body as the feature input.
    # Individual title segments are used only for rule-based evidence annotation.
    combined = title + "\n" + text[:5000]

    try:
        X_word = vec_w.transform([combined])
        X_char = vec_c.transform([combined])
        X_bin  = compute_binary_features(combined)

        from scipy.sparse import csr_matrix
        X_bin_sparse = csr_matrix(X_bin)
        X = hstack([X_word, X_char, X_bin_sparse])

        # Predict probabilities
        proba_matrix = clf.predict_proba(X)   # shape (1, n_labels)
        proba_row = proba_matrix[0]

        labels = lb.classes_
        probs: dict[str, float] = {
            lbl: float(proba_row[i]) for i, lbl in enumerate(labels)
        }
        matched = [lbl for lbl, p in probs.items() if p >= threshold]

        # Evidence: first regex hit per matched type
        evidence: list[DocTypeEvidence] = []
        for doc_type in matched:
            for pat_str in DOCTYPE_RULES.get(doc_type, []):
                m = re.search(pat_str, combined[:3000], re.IGNORECASE)
                if m:
                    evidence.append(DocTypeEvidence(
                        source="model",
                        page=1,
                        span_text=m.group(0),
                        char_start=m.start(),
                        char_end=m.end(),
                        rule_id=f"model+rule:{doc_type}",
                    ))
                    break
            else:
                evidence.append(DocTypeEvidence(
                    source="model",
                    page=1,
                    span_text=combined[:80],
                    char_start=0,
                    char_end=min(80, len(combined)),
                    rule_id=f"model:{doc_type}",
                ))

        matched, probs, evidence = _ensure_letter_base_tag(matched, probs, evidence)

        confidence = max(probs.values()) if probs else 0.0
        return DocTypeResult(
            document_types=matched,
            probabilities=probs,
            confidence=confidence,
            evidence=evidence,
            fallback_used=False,
        )
    except Exception:
        return _rule_based_classify(text, title)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document_type(
    text: str,
    title: str = "",
    threshold: float = DOCTYPE_THRESHOLD,
) -> DocTypeResult:
    """
    Classify document type(s) from text and optional title.

    Uses trained model if available; falls back to rule-based detection.
    Always returns multi-label output with probabilities and evidence.
    """
    return _model_classify(text, title, threshold)
