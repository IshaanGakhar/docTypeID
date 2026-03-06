"""
Court name, court location, and judge name extraction.
Primary: regex patterns on caption/title zone.
Secondary: CRF NER entities when model is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.config import (
    COURT_PATTERNS,
    COURT_LOCATION_PATTERNS,
    JUDGE_PATTERNS,
    US_STATES,
)
from pipeline.preprocess import DocumentZones, IndexedLine, lines_to_text
from pipeline.crf_ner import CRFResult, get_entities_by_label


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class CourtEvidence:
    source: str
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class CourtResult:
    court_name: Optional[str]
    court_location: Optional[str]
    judge_name: Optional[str]
    court_confidence: float
    location_confidence: float
    judge_confidence: float
    court_evidence: list[CourtEvidence]
    location_evidence: list[CourtEvidence]
    judge_evidence: list[CourtEvidence]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COMPILED_COURT = [re.compile(p, re.IGNORECASE) for p in COURT_PATTERNS]
_COMPILED_LOC   = [re.compile(p, re.IGNORECASE) for p in COURT_LOCATION_PATTERNS]
_COMPILED_JUDGE = [re.compile(p, re.IGNORECASE) for p in JUDGE_PATTERNS]
_COMPILED_STATE = re.compile(r"\b(" + US_STATES + r")\b", re.IGNORECASE)

_LOCATION_PREFIXES = re.compile(
    r"^(?:FOR\s+THE|IN\s+THE|OF\s+THE|OF)\s+", re.IGNORECASE
)

_COURT_NOISE = re.compile(
    r"\s*(?:,|\-)\s*(?:Division|Section|Branch|Department)\s*.*$", re.IGNORECASE
)

# Words that cannot start a valid judge name (structural/jurisdictional words).
# Prevents fragments like "for the\nDistrict" being classified as a judge name.
_JUDGE_BAD_STARTS = re.compile(
    r"^(?:for|the|of|in|and|or|district|court|division|section|"
    r"northern|southern|eastern|western|central|middle|california|texas|"
    r"new|york|florida|illinois|ohio)\b",
    re.IGNORECASE,
)
# Words that must not appear anywhere in a judge name
_JUDGE_STRUCTURAL = re.compile(
    r"\b(?:district|court|division|section|branch|jurisdiction)\b",
    re.IGNORECASE,
)


def _is_valid_judge_name(name: str) -> bool:
    """Return True only if name plausibly identifies a person."""
    name = name.strip()
    if not name or len(name) < 4:
        return False
    words = name.split()
    # First word must begin with an uppercase letter
    if not words[0][0].isupper():
        return False
    if _JUDGE_BAD_STARTS.match(name):
        return False
    if _JUDGE_STRUCTURAL.search(name):
        return False
    return True


def _clean_court_name(raw: str) -> str:
    raw = raw.strip()
    # Remove trailing noise like ", Division 3"
    raw = _COURT_NOISE.sub("", raw)
    # Collapse whitespace
    return re.sub(r"\s+", " ", raw).strip()


_LOC_LOWERCASE_WORDS = frozenset({"of", "the", "and", "or", "in", "at", "for", "de"})


def _clean_location(raw: str) -> str:
    raw = _LOCATION_PREFIXES.sub("", raw.strip())
    raw = re.sub(r"\s+", " ", raw).strip()
    if raw.isupper() or raw == raw.upper():
        # Title-case but keep prepositions/articles lowercase
        # e.g. "COUNTY OF SANTA CLARA" → "County of Santa Clara"
        words = raw.title().split()
        raw = " ".join(
            w.lower() if w.lower() in _LOC_LOWERCASE_WORDS and i > 0 else w
            for i, w in enumerate(words)
        )
    return raw


def _extract_court_regex(text: str) -> tuple[Optional[str], Optional[CourtEvidence]]:
    for i, pat in enumerate(_COMPILED_COURT):
        m = pat.search(text)
        if m:
            raw = m.group(1)
            name = _clean_court_name(raw)
            ev = CourtEvidence(
                source="regex",
                page=1,
                span_text=m.group(0),
                char_start=m.start(),
                char_end=m.end(),
                rule_id=f"court_pattern:{i}",
            )
            return name, ev
    return None, None


def _extract_location_regex(text: str) -> tuple[Optional[str], Optional[CourtEvidence]]:
    for i, pat in enumerate(_COMPILED_LOC):
        m = pat.search(text)
        if m:
            raw = m.group(1)
            loc = _clean_location(raw)
            ev = CourtEvidence(
                source="regex",
                page=1,
                span_text=m.group(0),
                char_start=m.start(),
                char_end=m.end(),
                rule_id=f"location_pattern:{i}",
            )
            return loc, ev

    # Fallback: extract a US state name
    m = _COMPILED_STATE.search(text)
    if m:
        return m.group(1).title(), CourtEvidence(
            source="regex",
            page=1,
            span_text=m.group(0),
            char_start=m.start(),
            char_end=m.end(),
            rule_id="state_fallback",
        )
    return None, None


def _extract_judge_regex(text: str) -> tuple[Optional[str], Optional[CourtEvidence]]:
    for i, pat in enumerate(_COMPILED_JUDGE):
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            if len(raw) < 3 or raw.upper() in ("THE", "HON", "A", "AN"):
                continue
            # Remove trailing noise (trailing comma-phrase and newlines)
            name = re.sub(r"\s*,.*$", "", raw).strip()
            name = name.split("\n")[0].strip()  # take only the first line
            if not _is_valid_judge_name(name):
                continue
            ev = CourtEvidence(
                source="regex",
                page=1,
                span_text=m.group(0),
                char_start=m.start(),
                char_end=m.end(),
                rule_id=f"judge_pattern:{i}",
            )
            return name, ev
    return None, None


def _page_of_match(char_offset: int, zones: DocumentZones) -> int:
    """Approximate page number for a character offset in zone text."""
    pos = 0
    for il in zones.all_lines:
        if pos + len(il.text) >= char_offset:
            return il.page_num
        pos += len(il.text) + 1
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_court_and_judge(
    zones: DocumentZones,
    crf_result: Optional[CRFResult] = None,
) -> CourtResult:
    """
    Extract court name, court location, and judge name.

    Searches caption_zone + title_zone first, then first_page_zone.
    Uses CRF entities as a confirmation / supplement when available.
    """
    # Build search text from caption zone + title zone (first 60 lines)
    caption_text = lines_to_text(zones.caption_zone)
    first_page_text = lines_to_text(zones.first_page_zone)
    search_text = caption_text + "\n" + first_page_text

    # --- Court name ---
    court_name, court_ev = _extract_court_regex(search_text)
    court_evidence: list[CourtEvidence] = []
    if court_ev:
        court_evidence.append(court_ev)

    # CRF supplement
    crf_court: Optional[str] = None
    if crf_result:
        crf_courts = get_entities_by_label(crf_result, "COURT")
        if crf_courts:
            crf_court = crf_courts[0]
            if not court_name:
                court_name = crf_court
                court_evidence.append(CourtEvidence(
                    source="crf", page=1, span_text=crf_court,
                    char_start=0, char_end=len(crf_court), rule_id="crf:COURT",
                ))

    court_confidence = 0.9 if court_name and court_ev else (0.6 if court_name else 0.0)

    # --- Court location ---
    court_location, loc_ev = _extract_location_regex(search_text)
    location_evidence: list[CourtEvidence] = []
    if loc_ev:
        location_evidence.append(loc_ev)

    # Try to extract location from court name if needed
    if not court_location and court_name:
        m = re.search(
            r"(?:FOR\s+THE|IN\s+THE|OF\s+THE)\s+((?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)?\s*DISTRICT\s+OF\s+[A-Z][A-Za-z\s]+?)(?:\n|$)",
            court_name + "\n" + search_text,
            re.IGNORECASE,
        )
        if m:
            court_location = _clean_location(m.group(1))
            location_evidence.append(CourtEvidence(
                source="regex", page=1, span_text=m.group(0),
                char_start=m.start(), char_end=m.end(),
                rule_id="location_from_court",
            ))

    location_confidence = 0.85 if court_location and loc_ev else (0.5 if court_location else 0.0)

    # --- Judge name ---
    judge_name, judge_ev = _extract_judge_regex(search_text)
    judge_evidence: list[CourtEvidence] = []
    if judge_ev:
        judge_evidence.append(judge_ev)

    # CRF supplement
    if crf_result:
        crf_judges = get_entities_by_label(crf_result, "JUDGE")
        if crf_judges:
            if not judge_name:
                judge_name = crf_judges[0]
                judge_evidence.append(CourtEvidence(
                    source="crf", page=1, span_text=crf_judges[0],
                    char_start=0, char_end=len(crf_judges[0]), rule_id="crf:JUDGE",
                ))

    judge_confidence = 0.85 if judge_name and judge_ev else (0.55 if judge_name else 0.0)

    return CourtResult(
        court_name=court_name,
        court_location=court_location,
        judge_name=judge_name,
        court_confidence=court_confidence,
        location_confidence=location_confidence,
        judge_confidence=judge_confidence,
        court_evidence=court_evidence,
        location_evidence=location_evidence,
        judge_evidence=judge_evidence,
    )
