"""
NUID (normalized unique identifier) extraction for legal documents.

Adapts the provided NUID extraction logic:
- Hierarchical pattern tiers (labeled → header → fallback → bare)
- Dash normalization (en/em dash → hyphen)
- Docket format validation
- Evidence spans
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.config import MIN_CHARS


# ---------------------------------------------------------------------------
# Pattern tiers (verbatim from provided extractor, reorganized for import)
# ---------------------------------------------------------------------------

BODY_DOCKET_PATTERNS = [
    r"(?:CIVIL ACTION|Civil Action)\s+No\.[:.]?\s*(\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"(?:CIVIL ACTION|Civil Action)\s+No\.[:.]?\s*(\d{2,4}-\d{3,6})",
    r"(?:CRIMINAL ACTION|Criminal Action)\s+No\.[:.]?\s*(\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    # "Case No.", "Case. No." (period misplaced after Case), "Civ. No." — all variants
    r"(?:Case\.?\s*No\.|Civ\.?\s*No\.)[:.]?\s*(\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"(?:Case\.?\s*No\.|Civ\.?\s*No\.)[:.]?\s*(\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"(?:CASE\s+NO\.|Case\s+No\.)[:.]?\s*([A-Z]{2,3}\s+\d{2,4}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"\bNo\.[:.]?\s*([A-Z]{2,3}-\d{2,4}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"MDL\s+(?:No\.|Docket\s+No\.)?\s*(\d{1,2}:\d{2,4}-md-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"MDL\s+(?:No\.|Docket\s+No\.)?\s*(\d{4,6})",
    r"(?:Docket\s+No\.|Dkt\.?\s*No\.)[:.]?\s*(\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"(?:Docket\s+No\.|Dkt\.?\s*No\.)[:.]?\s*(\d{2,4}-[a-z]{2,3}-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"(?:Index\s+No\.|File\s+No\.)[:.]?\s*(\d{2,6}[-/]\d{2,6}(?:[-/][A-Z0-9]+)?)",
]

HEADER_DOCKET_PATTERNS = [
    r"\bCase\s+(\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{4,6}(?:-[A-Z0-9\-]+)?)",
    r"\bCase\s+(\d+[-:][a-z]{2,3}-\d{4,6})",
]

FALLBACK_DOCKET_PATTERNS = [
    r"\bNo\.\s+(\d{2,4}-[a-z]{2,3}-\d{3,6})",
]

BARE_DOCKET_PATTERNS = [
    r"\b(\d{1,2}:\d{2,4}-(?:cv|cr|md|bk|ap|mj|po|mc)-\d{4,6}(?:-[A-Z0-9\-]+)?)",
    r"\b(\d{2,4}-(?:cv|cr|md|bk|ap|mj|po|mc)-\d{3,6}(?:-[A-Z0-9\-]+)?)",
    r"\b([A-Z]{2,3}-\d{2,4}-\d{4,6}(?:-[A-Z0-9\-]+)?)",
]

_TIERS = [
    ("labeled",  BODY_DOCKET_PATTERNS),
    ("header",   HEADER_DOCKET_PATTERNS),
    ("fallback", FALLBACK_DOCKET_PATTERNS),
    ("bare",     BARE_DOCKET_PATTERNS),
]

# ---------------------------------------------------------------------------
# Validation (verbatim from provided extractor)
# ---------------------------------------------------------------------------

_VALID_DOCKET_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}-(?:cv|cr|md|bk|ap|mj|po|mc|cv)-\d{3,6}", re.IGNORECASE),
    re.compile(r"^\d{2}-(?:cv|cr|md|bk|ap|mj|po|mc|cv)-\d{3,6}", re.IGNORECASE),
    re.compile(r"^\d{4}-(?:cv|cr|md|bk|ap)-\d+", re.IGNORECASE),
    re.compile(r"^\d{4,6}$"),
    re.compile(r"^\d{2}-\d{4,6}$"),
    re.compile(r"^[A-Z]{2,3}[-\s]\d{2,4}-\d{3,6}", re.IGNORECASE),
    re.compile(r"^[A-Z]{1,3}\d{5,}$", re.IGNORECASE),
    re.compile(r"^\d{5,}-[A-Z]{2,4}$", re.IGNORECASE),
    re.compile(r"^\d{4}\s+WL\s+\d+$", re.IGNORECASE),
    re.compile(r"^\d{4}\s+U\.S\.", re.IGNORECASE),
    re.compile(r"^\d{2,6}[-/]\d{2,6}(?:[-/][A-Z0-9]+)?$", re.IGNORECASE),
]


def _validate_docket(docket: str) -> bool:
    d = docket.strip()
    if len(d) < 4 or len(d) > 50:
        return False
    if not any(c.isdigit() for c in d):
        return False
    return any(p.search(d) for p in _VALID_DOCKET_PATTERNS)


def _normalize_dashes(text: str) -> str:
    return text.replace("\u2013", "-").replace("\u2014", "-")


# ---------------------------------------------------------------------------
# NUID normalization hook
# ---------------------------------------------------------------------------

def normalize_uid(raw: str) -> str:
    """
    Normalize a raw extracted docket/case number.
    - Strip surrounding whitespace
    - Collapse internal whitespace to single space
    - Normalize dashes
    - Uppercase type codes (cv, cr, md …)
    """
    if not raw:
        return raw
    uid = _normalize_dashes(raw.strip())
    # normalize whitespace
    uid = re.sub(r"\s+", " ", uid)
    # uppercase case type segment (e.g. 2:18-cv-… → 2:18-CV-…)
    uid = re.sub(
        r"(\d+[:\-])([a-z]{2,3})(\-)",
        lambda m: m.group(1) + m.group(2).upper() + m.group(3),
        uid,
    )
    return uid


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class UIDEvidence:
    source: str
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class UIDResult:
    nuid: Optional[str]         # normalized UID
    raw: Optional[str]          # raw matched string
    source_tier: Optional[str]  # "labeled" | "header" | "fallback" | "bare"
    confidence: float
    evidence: list[UIDEvidence]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_nuid(
    full_text: str,
    pages: list | None = None,  # list of PageInfo (from pdf_loader)
) -> UIDResult:
    """
    Extract and normalize the primary case/docket number.

    Searches the first ~2 pages (or first 6000 chars) using hierarchical tiers.
    First valid match wins.

    Parameters
    ----------
    full_text : normalized full document text
    pages     : list of PageInfo objects (from pdf_loader.LoadedDocument.pages)
    """
    # Scope search to first 2 pages
    if pages and len(pages) >= 2:
        search_text = _normalize_dashes(pages[0].text + "\n" + pages[1].text)
    elif pages and len(pages) == 1:
        search_text = _normalize_dashes(pages[0].text)
    else:
        search_text = _normalize_dashes(full_text[:6000])

    for tier_name, patterns in _TIERS:
        for pat_str in patterns:
            m = re.search(pat_str, search_text, re.IGNORECASE)
            if m:
                raw = m.group(1)
                if _validate_docket(raw):
                    normalized = normalize_uid(raw)
                    # Determine page number of the match
                    page_num = _match_page(m.start(), pages)
                    confidence = _tier_confidence(tier_name)
                    return UIDResult(
                        nuid=normalized,
                        raw=raw,
                        source_tier=tier_name,
                        confidence=confidence,
                        evidence=[UIDEvidence(
                            source="regex",
                            page=page_num,
                            span_text=m.group(0),
                            char_start=m.start(),
                            char_end=m.end(),
                            rule_id=f"uid:{tier_name}:{pat_str[:40]}",
                        )],
                    )

    return UIDResult(nuid=None, raw=None, source_tier=None, confidence=0.0, evidence=[])


def _tier_confidence(tier: str) -> float:
    return {"labeled": 0.95, "header": 0.80, "fallback": 0.65, "bare": 0.50}.get(tier, 0.4)


def _match_page(char_offset: int, pages: list | None) -> int:
    """Determine which page a character offset falls on (1-indexed)."""
    if not pages:
        return 1
    pos = 0
    for page in pages:
        pos += page.char_count + 1  # +1 for the "\n" joining pages
        if char_offset < pos:
            return page.page_num
    return pages[-1].page_num if pages else 1
