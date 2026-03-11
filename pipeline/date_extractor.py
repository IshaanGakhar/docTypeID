"""
Filing date extraction using regex patterns + dateutil normalization.
Outputs ISO-8601 dates (YYYY-MM-DD).
Prefers dates preceded by filing-context keywords.

Tier hierarchy (highest confidence first):
  Tier -1: ECF header          → 0.99
  Tier -0.5: Filing stamp      → 0.97
  Tier 0: Anchored patterns    → 0.95
  Tier 1: Context-window scan  → 0.90
  Tier 2: Positional fallback  → 0.50 (first/last 10 lines)
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
    ECF_HEADER_PATTERN, DATE_NEGATIVE_PATTERNS,
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

_ECF_RE = re.compile(ECF_HEADER_PATTERN, re.IGNORECASE)

_FILED_STAMP_RE = re.compile(r"^\s*FILED\s*$", re.IGNORECASE)

_NEG_CONTEXT_PATS = [re.compile(p, re.IGNORECASE) for p in DATE_NEGATIVE_PATTERNS]
_NEG_WINDOW = 80


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
                inner = date_pat.strip(r"\b").lstrip(r"\b").rstrip(r"\b")
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


def _has_negative_context(text: str, match_start: int, match_end: int) -> bool:
    """Return True if the date sits inside or adjacent to a citation / noise pattern."""
    win_start = max(0, match_start - _NEG_WINDOW)
    win_end = min(len(text), match_end + _NEG_WINDOW)
    window = text[win_start:win_end]
    return any(p.search(window) for p in _NEG_CONTEXT_PATS)


def _page_of_offset(char_offset: int, zones: DocumentZones) -> int:
    pos = 0
    for il in zones.all_lines:
        if pos + len(il.text) >= char_offset:
            return il.page_num
        pos += len(il.text) + 1
    return 1


# ---------------------------------------------------------------------------
# ECF header detection (Tier -1)
# ---------------------------------------------------------------------------

def _try_ecf_header(zones: DocumentZones) -> Optional[DateResult]:
    """Scan page 1 for a federal ECF header — the most reliable date source."""
    fp_lines = zones.first_page_zone or []
    fp_text = lines_to_text(fp_lines)
    m = _ECF_RE.search(fp_text)
    if not m:
        return None
    parsed = _parse_iso(m.group(1))
    if not parsed:
        return None
    return DateResult(
        filing_date=parsed,
        confidence=0.99,
        evidence=[DateEvidence(
            source="regex",
            page=1,
            span_text=m.group(0),
            char_start=m.start(),
            char_end=m.end(),
            rule_id="ecf_header",
        )],
    )


# ---------------------------------------------------------------------------
# Filing stamp detection (Tier -0.5)
# ---------------------------------------------------------------------------

def _try_filing_stamp(zones: DocumentZones) -> Optional[DateResult]:
    """
    Detect a multi-line clerk filing stamp:
      FILED
      DALLAS COUNTY
      7/1/2019 10:39 AM
      FELICIA PITRE
      DISTRICT CLERK

    When a standalone "FILED" line is found, scan the next 3 lines for a date.
    Only considers first-page and last-page lines.
    """
    if not zones.all_lines:
        return None

    all_page_nums = sorted({il.page_num for il in zones.all_lines})
    target_pages = {all_page_nums[0]}
    if len(all_page_nums) > 1:
        target_pages.add(all_page_nums[-1])

    target_lines = [il for il in zones.all_lines if il.page_num in target_pages]

    for i, il in enumerate(target_lines):
        if not _FILED_STAMP_RE.match(il.text):
            continue
        for j in range(i + 1, min(i + 4, len(target_lines))):
            candidate_text = target_lines[j].text.strip()
            for pat in _DATE_PATS:
                m = pat.search(candidate_text)
                if m:
                    parsed = _parse_iso(m.group(1))
                    if parsed:
                        stamp_span = il.text.strip() + " … " + candidate_text
                        return DateResult(
                            filing_date=parsed,
                            confidence=0.97,
                            evidence=[DateEvidence(
                                source="regex",
                                page=il.page_num,
                                span_text=stamp_span,
                                char_start=0,
                                char_end=len(stamp_span),
                                rule_id="filing_stamp",
                            )],
                        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_filing_date(zones: DocumentZones) -> DateResult:
    """
    Extract the filing date from document zones.

    Strategy (six tiers, highest confidence first):
    Tier -1: ECF header — "Case ... Filed MM/DD/YY ..."  → 0.99
    Tier -0.5: Filing stamp — standalone "FILED" + date nearby → 0.97
    Tier 0: Anchored patterns — keyword+separator+date     → 0.95
    Tier 1: Context-window — bare date with keyword nearby → 0.90
    Tier 2: Positional fallback — first/last 10 lines only → 0.50

    All tiers (except ECF and stamp) apply a negative-context filter to
    reject dates embedded in case citations, biographical mentions, etc.

    Candidates are sorted by (tier, sub_tier, format_index, char_offset).
    """

    # --- Tier -1: ECF header (authoritative, return immediately) ---
    ecf = _try_ecf_header(zones)
    if ecf:
        return ecf

    # --- Tier -0.5: Filing stamp (very reliable, return immediately) ---
    stamp = _try_filing_stamp(zones)
    if stamp:
        return stamp

    # --- Tiers 0, 1, 2 ---
    # candidates: (parsed_date, tier, sub_tier, fmt_idx, match, page_num)
    candidates: list[tuple[str, int, int, int, re.Match, int]] = []

    all_page_nums_ks = sorted({il.page_num for il in zones.all_lines}) if zones.all_lines else []
    last_page_lines_ks = (
        [il for il in zones.all_lines if il.page_num == all_page_nums_ks[-1]]
        if all_page_nums_ks else []
    )

    keyword_zones = [
        zones.caption_zone,
        zones.first_page_zone,
        zones.title_zone,
        last_page_lines_ks,
    ]

    for zone_lines in keyword_zones:
        zone_text = lines_to_text(zone_lines)

        # Tier 0 — anchored keyword + separator + date
        for anc_pat, _stem, fmt_idx in _ANCHORED_PATS:
            for m in anc_pat.finditer(zone_text):
                if _has_negative_context(zone_text, m.start(), m.end()):
                    continue
                parsed = _parse_iso(m.group(1))
                if parsed:
                    page_num = _page_of_offset(m.start(), zones)
                    candidates.append((parsed, 0, 0, fmt_idx, m, page_num))

        # Tier 1 — bare date with a keyword in the preceding 80-char window
        for fmt_idx, pat in enumerate(_DATE_PATS):
            for m in pat.finditer(zone_text):
                if _has_negative_context(zone_text, m.start(), m.end()):
                    continue
                if _has_context(zone_text, m.start()):
                    parsed = _parse_iso(m.group(1))
                    if parsed:
                        page_num = _page_of_offset(m.start(), zones)
                        candidates.append((parsed, 1, 0, fmt_idx, m, page_num))

    # Tier 2 — positional: restricted to first/last 10 lines, short lines only
    if zones.all_lines:
        all_page_nums = sorted({il.page_num for il in zones.all_lines})
        first_pn = all_page_nums[0]
        last_pn  = all_page_nums[-1]
        single_page = (first_pn == last_pn)

        fp_lines = [il for il in zones.all_lines if il.page_num == first_pn]
        lp_lines = fp_lines if single_page else [
            il for il in zones.all_lines if il.page_num == last_pn
        ]

        fp_head = fp_lines[:10]
        lp_tail = lp_lines[-10:] if len(lp_lines) >= 10 else lp_lines

        if single_page:
            pos_zones: list[tuple[int, list, int]] = [
                (0, fp_head, first_pn),
                (1, lp_tail, first_pn),
            ]
        else:
            pos_zones = [
                (0, fp_head, first_pn),
                (1, lp_tail, last_pn),
            ]

        for sub_tier, pz_lines, page_num in pos_zones:
            if not pz_lines:
                continue
            zone_text = lines_to_text(pz_lines)
            for fmt_idx, pat in enumerate(_DATE_PATS):
                for m in pat.finditer(zone_text):
                    line_text = m.group(0)
                    line_ctx = zone_text[max(0, m.start() - 10):m.end() + 10]
                    if len(line_ctx.strip()) > 80:
                        continue
                    if _has_negative_context(zone_text, m.start(), m.end()):
                        continue
                    parsed = _parse_iso(m.group(1))
                    if parsed:
                        candidates.append((parsed, 2, sub_tier, fmt_idx, m, page_num))

    if not candidates:
        return DateResult(filing_date=None, confidence=0.0, evidence=[])

    # Sort: tier → sub_tier → format index → char offset
    candidates.sort(key=lambda c: (c[1], c[2], c[3], c[4].start()))

    parsed_date, tier, sub_tier, fmt_idx, m, page_num = candidates[0]

    confidence = {0: 0.95, 1: 0.90, 2: 0.50}[tier]
    _pos_labels = {0: "fp_first_half", 1: "lp_second_half", 2: "fp_lp_rest"}
    tier_label  = {0: "anchored", 1: "contextual", 2: f"positional:{_pos_labels.get(sub_tier, 'other')}"}[tier]

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
