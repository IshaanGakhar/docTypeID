"""
Filing date extraction using regex patterns + dateutil normalization.
Outputs ISO-8601 dates (YYYY-MM-DD).
Prefers dates preceded by filing-context keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

from pipeline.config import (
    DATE_PATTERNS, DATE_CONTEXT_KEYWORDS,
    DATE_PREFIX_STEMS, DATE_PREFIX_SEPARATORS,
)
from pipeline.preprocess import DocumentZones, lines_to_text


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class DateEvidence:
    source: str
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class DateResult:
    filing_date: Optional[str]   # ISO-8601 "YYYY-MM-DD" or None
    confidence: float
    evidence: list[DateEvidence]


# ---------------------------------------------------------------------------
# Pattern compilation
# ---------------------------------------------------------------------------

_DATE_PATS    = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]
_CONTEXT_PATS = [re.compile(p, re.IGNORECASE) for p in DATE_CONTEXT_KEYWORDS]
_CONTEXT_WINDOW = 80  # chars before date match to check for context keyword


def _build_anchored_patterns() -> list[tuple[re.Pattern, str, int]]:
    """
    Build combined prefix+separator+date patterns covering every combination of:
      - keyword stem  (filed, dated, date, filing date, …)
      - separator     (: / - / – / — / space)
      - date format   (all DATE_PATTERNS)

    Returns list of (compiled_pattern, keyword_stem, date_format_index).
    The date capture group is always group(1).
    """
    anchored: list[tuple[re.Pattern, str, int]] = []
    for stem in DATE_PREFIX_STEMS:
        for sep in DATE_PREFIX_SEPARATORS:
            for fmt_idx, date_pat in enumerate(DATE_PATTERNS):
                # Strip the outer \b…\b from date patterns so we can embed them
                inner = date_pat.strip(r"\b").lstrip(r"\b").rstrip(r"\b")
                # Full anchored pattern: keyword + separator + date (date group = 1)
                full = rf"(?:{stem}){sep}({inner})"
                try:
                    anchored.append((
                        re.compile(full, re.IGNORECASE),
                        stem,
                        fmt_idx,
                    ))
                except re.error:
                    pass
    return anchored


_ANCHORED_PATS = _build_anchored_patterns()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(raw: str) -> Optional[str]:
    """Parse an arbitrary date string and return YYYY-MM-DD, or None on failure."""
    try:
        dt = dateutil_parser.parse(raw, dayfirst=False, yearfirst=False)
        if dt.year < 1900 or dt.year > 2100:
            return None
        return dt.strftime("%Y-%m-%d")
    except (ParserError, OverflowError, ValueError):
        return None


def _has_context(text: str, match_start: int) -> bool:
    """Return True if a filing-context keyword appears just before the date."""
    window_start = max(0, match_start - _CONTEXT_WINDOW)
    window = text[window_start:match_start]
    return any(p.search(window) for p in _CONTEXT_PATS)


def _page_of_offset(char_offset: int, zones: DocumentZones) -> int:
    pos = 0
    for il in zones.all_lines:
        if pos + len(il.text) >= char_offset:
            return il.page_num
        pos += len(il.text) + 1
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_filing_date(zones: DocumentZones) -> DateResult:
    """
    Extract the filing date from document zones.

    Strategy (three tiers, highest confidence first):
    1. Anchored patterns — keyword + separator + date in one regex.
       e.g. "Filed: Jan 5, 2019", "Date - 01/05/2019", "Dated 2019-01-05"
       → confidence 0.95
    2. Context-window scan — bare date preceded by a keyword within 80 chars,
       in caption / first-page / title zones.
       → confidence 0.90
    3. Positional fallback — any bare date on the first or last page,
       with three sub-tiers ranked by location:
         sub 0: first half of page 1          (most likely filing stamp)
         sub 1: second half of last page      (most likely signature block)
         sub 2: rest of page 1 or last page
       → confidence 0.65

    Candidates are sorted by (tier, sub_tier, format_index, char_offset).
    """
    # candidates: (parsed_date, tier, sub_tier, fmt_idx, match, page_num)
    # For tiers 0 and 1, sub_tier is always 0 (unused for ranking).
    candidates: list[tuple[str, int, int, int, re.Match, int]] = []

    # Build keyword-scan zones: front matter + last page (signature blocks live there)
    all_page_nums_ks = sorted({il.page_num for il in zones.all_lines}) if zones.all_lines else []
    last_page_lines_ks = (
        [il for il in zones.all_lines if il.page_num == all_page_nums_ks[-1]]
        if all_page_nums_ks else []
    )

    keyword_zones = [
        zones.caption_zone,
        zones.first_page_zone,
        zones.title_zone,
        last_page_lines_ks,   # "Dated: …" signature lines live here
    ]

    for zone_lines in keyword_zones:
        zone_text = lines_to_text(zone_lines)

        # Tier 0 — anchored keyword + separator + date
        for anc_pat, _stem, fmt_idx in _ANCHORED_PATS:
            for m in anc_pat.finditer(zone_text):
                parsed = _parse_iso(m.group(1))
                if parsed:
                    page_num = _page_of_offset(m.start(), zones)
                    candidates.append((parsed, 0, 0, fmt_idx, m, page_num))

        # Tier 1 — bare date with a keyword in the preceding 80-char window
        for fmt_idx, pat in enumerate(_DATE_PATS):
            for m in pat.finditer(zone_text):
                if _has_context(zone_text, m.start()):
                    parsed = _parse_iso(m.group(1))
                    if parsed:
                        page_num = _page_of_offset(m.start(), zones)
                        candidates.append((parsed, 1, 0, fmt_idx, m, page_num))

    # Tier 2 — positional: first and last page, split into halves
    if zones.all_lines:
        all_page_nums = sorted({il.page_num for il in zones.all_lines})
        first_pn = all_page_nums[0]
        last_pn  = all_page_nums[-1]
        single_page = (first_pn == last_pn)

        fp_lines = [il for il in zones.all_lines if il.page_num == first_pn]
        lp_lines = fp_lines if single_page else [
            il for il in zones.all_lines if il.page_num == last_pn
        ]

        fp_mid = max(1, len(fp_lines) // 2)
        lp_mid = max(1, len(lp_lines) // 2)

        if single_page:
            # First half = sub 0 (filing stamp area),
            # second half = sub 1 (signature / date block)
            pos_zones: list[tuple[int, list, int]] = [
                (0, fp_lines[:fp_mid], first_pn),
                (1, fp_lines[fp_mid:], first_pn),
            ]
        else:
            pos_zones = [
                (0, fp_lines[:fp_mid],  first_pn),  # first half of page 1
                (1, lp_lines[lp_mid:],  last_pn),   # second half of last page
                (2, fp_lines[fp_mid:],  first_pn),  # second half of page 1
                (2, lp_lines[:lp_mid],  last_pn),   # first half of last page
            ]

        for sub_tier, pz_lines, page_num in pos_zones:
            if not pz_lines:
                continue
            zone_text = lines_to_text(pz_lines)
            for fmt_idx, pat in enumerate(_DATE_PATS):
                for m in pat.finditer(zone_text):
                    parsed = _parse_iso(m.group(1))
                    if parsed:
                        candidates.append((parsed, 2, sub_tier, fmt_idx, m, page_num))

    if not candidates:
        return DateResult(filing_date=None, confidence=0.0, evidence=[])

    # Sort: tier → sub_tier → format index → char offset
    candidates.sort(key=lambda c: (c[1], c[2], c[3], c[4].start()))

    parsed_date, tier, sub_tier, fmt_idx, m, page_num = candidates[0]

    confidence = {0: 0.95, 1: 0.90, 2: 0.65}[tier]
    _pos_labels = {0: "fp_first_half", 1: "lp_second_half", 2: "fp_lp_rest"}
    tier_label  = {0: "anchored", 1: "contextual", 2: f"positional:{_pos_labels[sub_tier]}"}[tier]

    evidence = [DateEvidence(
        source="regex",
        page=page_num,
        span_text=m.group(0),
        char_start=m.start(),
        char_end=m.end(),
        rule_id=f"date_pattern:{fmt_idx}:{tier_label}",
    )]

    return DateResult(
        filing_date=parsed_date,
        confidence=confidence,
        evidence=evidence,
    )
