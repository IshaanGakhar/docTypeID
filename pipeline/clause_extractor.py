"""
Constitutional clause / provision citation detector.

Scans document text for references to US Constitutional provisions:
  - Amendments (First through Twenty-Seventh, ordinal and roman numeral forms)
  - Named clauses (Due Process, Commerce, Equal Protection, etc.)
  - Article / Section citations (Art. I § 8, Article II Section 2, etc.)

Each hit is deduplicated by canonical name and returned with a short
context snippet showing where in the document it was cited.  The result
list is intended as a "constitutional fingerprint" for a legal document —
useful when drafting similar filings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.preprocess import DocumentZones, IndexedLine


# ---------------------------------------------------------------------------
# Data classes (same external API as before so run_pipeline.py needs no change)
# ---------------------------------------------------------------------------

@dataclass
class ClauseEvidence:
    source: str
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class ClauseItem:
    clause_type: str          # "amendment" | "named_clause" | "article_section"
    heading: str              # canonical provision name
    text: str                 # context sentence(s) where it was found
    page_start: int
    page_end: int
    evidence: list[ClauseEvidence] = field(default_factory=list)


@dataclass
class ClauseResult:
    clauses: list[ClauseItem]
    evidence: list[ClauseEvidence]


# ---------------------------------------------------------------------------
# Amendment name tables
# ---------------------------------------------------------------------------

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27,
}

_NUMERIC_SUFFIX = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
    6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
    11: "11th", 12: "12th", 13: "13th", 14: "14th", 15: "15th",
    16: "16th", 17: "17th", 18: "18th", 19: "19th", 20: "20th",
    21: "21st", 22: "22nd", 23: "23rd", 24: "24th", 25: "25th",
    26: "26th", 27: "27th",
}

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20, "XXI": 21, "XXII": 22, "XXIII": 23,
    "XXIV": 24, "XXV": 25, "XXVI": 26, "XXVII": 27,
}

def _canon_amendment(n: int) -> str:
    return f"{_NUMERIC_SUFFIX[n]} Amendment"


# ---------------------------------------------------------------------------
# Named clause catalogue
# Keyed by canonical name → (clause_type, list of regex patterns)
# ---------------------------------------------------------------------------

_NAMED_CLAUSES: list[tuple[str, str, str]] = [
    # (canonical_name, clause_type, regex_pattern)

    # --- Widely litigated named clauses ---
    ("Due Process Clause",            "named_clause",
     r"due\s+process(?:\s+clause|\s+of\s+law)?"),
    ("Equal Protection Clause",       "named_clause",
     r"equal\s+protection(?:\s+clause)?"),
    ("Commerce Clause",               "named_clause",
     r"(?:dormant\s+)?commerce\s+clause"),
    ("Supremacy Clause",              "named_clause",
     r"supremacy\s+clause"),
    ("Establishment Clause",          "named_clause",
     r"establishment\s+clause"),
    ("Free Exercise Clause",          "named_clause",
     r"free\s+exercise\s+clause"),
    ("Free Speech Clause",            "named_clause",
     r"free(?:dom of)?\s+speech(?:\s+clause)?"),
    ("Takings Clause",                "named_clause",
     r"takings?\s+clause|just\s+compensation\s+clause"),
    ("Confrontation Clause",          "named_clause",
     r"confrontation\s+clause"),
    ("Self-Incrimination Clause",     "named_clause",
     r"self[- ]incrimination\s+(?:clause|privilege)|privilege\s+against\s+self[- ]incrimination"),
    ("Double Jeopardy Clause",        "named_clause",
     r"double\s+jeopardy(?:\s+clause)?"),
    ("Cruel and Unusual Punishment",  "named_clause",
     r"cruel\s+and\s+unusual\s+punish(?:ment|es)"),
    ("Privileges and Immunities Clause", "named_clause",
     r"privileges?\s+and\s+immunities?\s+clause"),
    ("Full Faith and Credit Clause",  "named_clause",
     r"full\s+faith\s+and\s+credit(?:\s+clause)?"),
    ("Necessary and Proper Clause",   "named_clause",
     r"necessary\s+and\s+proper\s+clause|elastic\s+clause"),
    ("Contracts Clause",              "named_clause",
     r"contracts?\s+clause|obligation\s+of\s+contracts?"),
    ("Ex Post Facto Clause",          "named_clause",
     r"ex\s+post\s+facto(?:\s+clause)?"),
    ("Appointments Clause",           "named_clause",
     r"appointments?\s+clause"),
    ("Presentment Clause",            "named_clause",
     r"presentment\s+clause"),
    ("Vesting Clause",                "named_clause",
     r"vesting\s+clause"),
    ("Speech or Debate Clause",       "named_clause",
     r"speech\s+or\s+debate\s+clause"),
    ("Origination Clause",            "named_clause",
     r"origination\s+clause"),
    ("Emoluments Clause",             "named_clause",
     r"emoluments?\s+clause"),
    ("General Welfare Clause",        "named_clause",
     r"general\s+welfare(?:\s+clause)?"),
    ("Spending Clause",               "named_clause",
     r"spending\s+clause"),
    ("War Powers",                    "named_clause",
     r"war\s+powers(?:\s+clause|\s+resolution)?"),
    ("Right to Bear Arms",            "named_clause",
     r"right\s+to\s+(?:keep\s+and\s+)?bear\s+arms"),
    ("Right to Counsel",              "named_clause",
     r"right\s+to\s+counsel"),
    ("Right to Speedy Trial",         "named_clause",
     r"(?:right\s+to\s+a?\s+)?speedy\s+trial(?:\s+clause)?"),
    ("Freedom of Assembly",           "named_clause",
     r"freedom\s+of\s+(?:peaceful\s+)?assembly"),
    ("Freedom of the Press",          "named_clause",
     r"freedom\s+of\s+(?:the\s+)?press"),
    ("Freedom of Religion",           "named_clause",
     r"freedom\s+of\s+religion"),
    ("Search and Seizure",            "named_clause",
     r"(?:unreasonable\s+)?search(?:es)?\s+and\s+seizures?"),
    ("Petition Clause",               "named_clause",
     r"right\s+to\s+petition(?:\s+(?:the\s+)?government)?"),
]

# Compile all named-clause patterns
_NAMED_COMPILED: list[tuple[str, str, re.Pattern]] = [
    (canon, ctype, re.compile(pattern, re.IGNORECASE))
    for canon, ctype, pattern in _NAMED_CLAUSES
]

# ---------------------------------------------------------------------------
# Amendment patterns
# ---------------------------------------------------------------------------

# Word-ordinal: "First Amendment", "Fourteenth Amendment"
_ORDINAL_AMEND = re.compile(
    r"\b(" + "|".join(re.escape(o) for o in _ORDINALS) + r")\s+amendment\b",
    re.IGNORECASE,
)

# Numeric: "1st Amendment", "14th Amend.", "42nd Amendment"
_NUMERIC_AMEND = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)\.?\s+amend(?:ment)?\.?\b",
    re.IGNORECASE,
)

# Roman numeral: "Amendment I", "Amendment XIV", "Amend. XIV"
_ROMAN_AMEND = re.compile(
    r"\bamend(?:ment)?\.?\s+(X{0,3}(?:IX|IV|V?I{0,3}))\b",
    re.IGNORECASE,
)

# U.S. Const. amend. XIV  /  U.S.C. Amend. IV
_CONST_AMEND = re.compile(
    r"u\.?s\.?\s+const(?:itution)?\.?\s+amend(?:ment)?\.?\s+(X{0,3}(?:IX|IV|V?I{0,3})|\d{1,2})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Article / Section patterns
# ---------------------------------------------------------------------------

# The US Constitution has exactly 7 articles (I–VII).
# Bare "Article I/II/..." is too ambiguous (contracts, bylaws, corporate docs
# all use numbered articles), so we only match when "U.S. Const." or
# "Constitution" appears explicitly nearby, OR when the article number
# is one of I–VII AND the surrounding text contains constitutional language.

# Requires explicit "U.S. Const." / "Constitution" prefix
_CONST_ARTICLE_RE = re.compile(
    r"(?:u\.?s\.?\s+const(?:itution)?|the\s+constitution)\.?\s+"
    r"(?:art(?:icle)?\.?\s+)?"
    r"(VII|VI|V|IV|III|II|I)"           # only Articles I–VII
    r"(?:\s*,?\s*(?:§|sec(?:tion)?\.?)\s*(\d{1,2}))?",
    re.IGNORECASE,
)

# "Article I, Section 8" bare — only accept Articles I–VII and only when
# the word "Constitution" appears within 300 characters
_BARE_ARTICLE_RE = re.compile(
    r"\b(?:article|art\.)\s+(VII|VI|V|IV|III|II|I)"
    r"(?:\s*,?\s*(?:section|sec\.|§)\s*(\d{1,2}))?",
    re.IGNORECASE,
)

_CONST_CONTEXT_RE = re.compile(
    r"constitution|u\.s\.\s+const|amendment|bill\s+of\s+rights",
    re.IGNORECASE,
)

_ROMAN_TO_INT = {v: k for k, v in _ROMAN.items()}  # not used but kept for clarity


# ---------------------------------------------------------------------------
# Context extraction helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _context_around(full_text: str, match_start: int, match_end: int,
                    window: int = 120) -> str:
    """Return up to `window` chars on each side of the match, trimmed."""
    start = max(0, match_start - window)
    end   = min(len(full_text), match_end + window)
    raw   = full_text[start:end].strip()
    # Collapse internal whitespace/newlines
    return " ".join(raw.split())


def _page_for_offset(char_offset: int, line_index: list[IndexedLine]) -> int:
    """Binary-search the line list to find the page number for a char offset."""
    lo, hi = 0, len(line_index) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_index[mid].char_start <= char_offset:
            lo = mid + 1
        else:
            hi = mid - 1
    idx = max(0, lo - 1)
    return line_index[idx].page_num if line_index else 1


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def _scan_text(
    full_text: str,
    line_index: list[IndexedLine],
) -> list[ClauseItem]:
    """
    Scan `full_text` for all constitutional citations.
    Returns a list of ClauseItems, deduplicated by canonical heading.
    """
    seen: dict[str, ClauseItem] = {}   # canonical name → first hit

    def _add(canon: str, ctype: str, m: re.Match) -> None:
        if canon in seen:
            return
        ctx  = _context_around(full_text, m.start(), m.end())
        page = _page_for_offset(m.start(), line_index)
        ev   = ClauseEvidence(
            source="regex",
            page=page,
            span_text=m.group(0),
            char_start=m.start(),
            char_end=m.end(),
            rule_id=f"const_cite:{ctype}",
        )
        seen[canon] = ClauseItem(
            clause_type=ctype,
            heading=canon,
            text=ctx,
            page_start=page,
            page_end=page,
            evidence=[ev],
        )

    # 1. Named clauses
    for canon, ctype, pat in _NAMED_COMPILED:
        for m in pat.finditer(full_text):
            _add(canon, ctype, m)

    # 2. Ordinal amendments ("Fourteenth Amendment")
    for m in _ORDINAL_AMEND.finditer(full_text):
        n = _ORDINALS[m.group(1).lower()]
        _add(_canon_amendment(n), "amendment", m)

    # 3. Numeric amendments ("14th Amendment")
    for m in _NUMERIC_AMEND.finditer(full_text):
        n = int(m.group(1))
        if 1 <= n <= 27:
            _add(_canon_amendment(n), "amendment", m)

    # 4. Roman-numeral amendments ("Amendment XIV")
    for m in _ROMAN_AMEND.finditer(full_text):
        roman = m.group(1).upper()
        if roman in _ROMAN:
            _add(_canon_amendment(_ROMAN[roman]), "amendment", m)

    # 5. U.S. Const. amend. XIV style
    for m in _CONST_AMEND.finditer(full_text):
        raw = m.group(1)
        n   = _ROMAN.get(raw.upper()) or (int(raw) if raw.isdigit() else None)
        if n and 1 <= n <= 27:
            _add(_canon_amendment(n), "amendment", m)

    # 6. Explicit "U.S. Const. art. I § 8" citations
    for m in _CONST_ARTICLE_RE.finditer(full_text):
        roman = m.group(1).upper()
        sec   = m.group(2)
        canon = f"Article {roman}" + (f", Section {sec}" if sec else "")
        _add(canon, "article_section", m)

    # 7. Bare "Article I/II/..." — only when constitutional context nearby
    for m in _BARE_ARTICLE_RE.finditer(full_text):
        roman = m.group(1).upper()
        sec   = m.group(2)
        # Check within 300 chars on either side for constitutional context
        window_start = max(0, m.start() - 300)
        window_end   = min(len(full_text), m.end() + 300)
        window       = full_text[window_start:window_end]
        if not _CONST_CONTEXT_RE.search(window):
            continue
        canon = f"Article {roman}" + (f", Section {sec}" if sec else "")
        _add(canon, "article_section", m)

    # Sort by first appearance (char_start of first evidence)
    return sorted(seen.values(), key=lambda c: c.evidence[0].char_start)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clauses(zones: DocumentZones) -> ClauseResult:
    """
    Detect US Constitutional citations in the document.

    Returns a ClauseResult whose `clauses` list is the deduplicated set of
    constitutional provisions (amendments, named clauses, article-section
    references) cited anywhere in the document, each with a context snippet.
    """
    # Build a flat text and a parallel line index for page-number lookup
    all_lines: list[IndexedLine] = (
        zones.caption_zone
        + zones.title_zone
        + zones.body_zone
    )
    if not all_lines:
        return ClauseResult(clauses=[], evidence=[])

    full_text  = "\n".join(il.text for il in all_lines)
    clauses    = _scan_text(full_text, all_lines)
    all_ev     = [ev for c in clauses for ev in c.evidence]

    return ClauseResult(clauses=clauses, evidence=all_ev)
