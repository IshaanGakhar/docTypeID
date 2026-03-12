"""
Legal citation and clause detector.

Scans document text for references to:
  - US Constitutional provisions (amendments, named clauses, article/section)
  - Federal statutes (U.S.C. §, named acts like RICO, ERISA, TCPA)
  - Federal rules (FRCP, FRE, FRAP, local rules)
  - State statutes (state code § citations)
  - Code of Federal Regulations (C.F.R.)
  - Case law citations (reporter-based, e.g. "556 U.S. 662")

Each hit is deduplicated by canonical name and returned with a short
context snippet showing where in the document it was cited.  The result
list is intended as a "legal fingerprint" for a document — useful when
drafting similar filings.
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
                              # | "federal_statute" | "state_statute"
                              # | "federal_rule" | "case_citation"
                              # | "named_act" | "cfr"
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
# Federal statute patterns  (Option A)
# ---------------------------------------------------------------------------

# § character: U+00A7.  PDFs sometimes render it as "Sec." or "Section".
_SEC = r"(?:§§?\s*|(?:Sec(?:tion)?\.?\s*))"

# Title N of the United States Code: "15 U.S.C. § 78j(b)", "42 U.S.C. §§ 1983, 1988"
_USC_RE = re.compile(
    r"\b(\d{1,2})\s+U\.?\s*S\.?\s*C\.?\s*"
    + _SEC
    + r"(\d[\w\-\.]*(?:\([a-zA-Z0-9]+\))*)"
    r"(?:\s*(?:,|and|&)\s*(\d[\w\-\.]*(?:\([a-zA-Z0-9]+\))*))?",
    re.IGNORECASE,
)

# Code of Federal Regulations: "17 C.F.R. § 240.10b-5"
_CFR_RE = re.compile(
    r"\b(\d{1,2})\s+C\.?\s*F\.?\s*R\.?\s*"
    + _SEC
    + r"(\d[\w\-\.]*(?:\([a-zA-Z0-9]+\))*)",
    re.IGNORECASE,
)

# Bare § with a title number preceding: "§ 1983", "§§ 10(b) and 20(a)"
# Only match when the preceding context looks statutory (title number or
# code name within 50 chars).
_BARE_SECTION_RE = re.compile(
    r"§§?\s*(\d[\w\-\.]*(?:\([a-zA-Z0-9]+\))*)",
)

# ---------------------------------------------------------------------------
# State statute patterns
# ---------------------------------------------------------------------------

# Generalized state code pattern:
#   [State abbrev / name] [Code name] § NNN
#   e.g. "Tex. Civ. Prac. & Rem. Code § 27.001"
#        "Cal. Code Civ. Proc. § 425.16"
#        "N.Y. Bus. Corp. Law § 720"
#        "Del. Code Ann. tit. 8, § 220"
_STATE_CODE_RE = re.compile(
    r"\b("
    # 2-4 letter state abbreviations with periods
    r"(?:[A-Z][a-z]{0,3}\.)\s+"
    # Code name: 1-6 words (letters, periods, ampersands)
    r"(?:[A-Z][A-Za-z&\.\']+\s+){0,5}"
    r"(?:Code|Law|Stat(?:utes)?|Ann(?:otated)?|Rev(?:ised)?|Gen(?:eral)?|Acts?)"
    r"(?:\s+(?:Ann(?:otated)?|Rev(?:ised)?|tit\.\s*\d+))?"
    r")"
    r"\s*(?:,\s*)?"
    + _SEC
    + r"(\d[\w\-\.]*(?:\([a-zA-Z0-9]+\))*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Federal Rules patterns
# ---------------------------------------------------------------------------

_FEDERAL_RULES: list[tuple[str, re.Pattern]] = [
    ("Fed. R. Civ. P.", re.compile(
        r"\bFed(?:eral)?\.?\s*R(?:ule)?\.?\s*(?:of\s+)?Civ(?:il)?\.?\s*P(?:roc(?:edure)?)?\.?"
        r"\s*(\d{1,3})"
        r"(?:\s*\(\s*([a-zA-Z0-9]+)\s*\))?",
        re.IGNORECASE,
    )),
    ("Fed. R. Evid.", re.compile(
        r"\bFed(?:eral)?\.?\s*R(?:ule)?\.?\s*(?:of\s+)?Evid(?:ence)?\.?"
        r"\s*(\d{1,4})",
        re.IGNORECASE,
    )),
    ("Fed. R. App. P.", re.compile(
        r"\bFed(?:eral)?\.?\s*R(?:ule)?\.?\s*(?:of\s+)?App(?:ellate)?\.?\s*P(?:roc(?:edure)?)?\.?"
        r"\s*(\d{1,3})",
        re.IGNORECASE,
    )),
    ("Fed. R. Bankr. P.", re.compile(
        r"\bFed(?:eral)?\.?\s*R(?:ule)?\.?\s*(?:of\s+)?Bankr(?:uptcy)?\.?\s*P(?:roc(?:edure)?)?\.?"
        r"\s*(\d{1,4})",
        re.IGNORECASE,
    )),
]

# "Rule 12(b)(6)", "Rule 23", "Rule 56" — bare rule references
# Only match when "Rule" is capitalized (avoids "golden rule", etc.)
_BARE_RULE_RE = re.compile(
    r"\bRule\s+(\d{1,3})"
    r"((?:\s*\(\s*[a-zA-Z0-9]+\s*\))+)?",
)

# Local Rules: "Local Rule 7.1", "L.R. 56.1", "Local Civ. R. 7.1"
_LOCAL_RULE_RE = re.compile(
    r"\b(?:Local\s+(?:Civ(?:il)?\.?\s*)?(?:Rule|R\.)|L\.?\s*(?:Civ\.?\s*)?R\.)"
    r"\s*(\d[\w\.\-]*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Named federal statutes / acts
# ---------------------------------------------------------------------------

_NAMED_ACTS: list[tuple[str, re.Pattern]] = [
    ("Securities Exchange Act of 1934", re.compile(
        r"\bSecurities\s+Exchange\s+Act(?:\s+of\s+1934)?\b", re.IGNORECASE)),
    ("Securities Act of 1933", re.compile(
        r"\bSecurities\s+Act(?:\s+of\s+1933)?\b", re.IGNORECASE)),
    ("Private Securities Litigation Reform Act", re.compile(
        r"\bP(?:rivate\s+)?S(?:ecurities\s+)?L(?:itigation\s+)?R(?:eform\s+)?A(?:ct)?\b"
        r"|PSLRA\b", re.IGNORECASE)),
    ("Sarbanes-Oxley Act", re.compile(
        r"\bSarbanes[- ]Oxley(?:\s+Act)?\b|SOX\b", re.IGNORECASE)),
    ("Dodd-Frank Act", re.compile(
        r"\bDodd[- ]Frank(?:\s+(?:Wall\s+Street\s+Reform\s+and\s+Consumer\s+Protection\s+)?Act)?\b",
        re.IGNORECASE)),
    ("RICO", re.compile(
        r"\bRICO\b|\bRacketeer\s+Influenced\s+and\s+Corrupt\s+Organizations?\b",
        re.IGNORECASE)),
    ("ERISA", re.compile(
        r"\bERISA\b|\bEmployee\s+Retirement\s+Income\s+Security\s+Act\b",
        re.IGNORECASE)),
    ("TCPA (Texas Citizens Participation Act)", re.compile(
        r"\bTexas\s+Citizens?\s+Participation\s+Act\b|"
        r"\bTex(?:as)?\.?\s+Civ(?:il)?\.?\s+Prac(?:tice)?\.?\s+(?:&|and)\s+Rem(?:edies)?\.?\s+Code\s+"
        + _SEC + r"27\b",
        re.IGNORECASE)),
    ("TCPA (Telephone Consumer Protection Act)", re.compile(
        r"\bTelephone\s+Consumer\s+Protection\s+Act\b", re.IGNORECASE)),
    ("FLSA", re.compile(
        r"\bFLSA\b|\bFair\s+Labor\s+Standards\s+Act\b", re.IGNORECASE)),
    ("FMLA", re.compile(
        r"\bFMLA\b|\bFamily\s+(?:and\s+)?Medical\s+Leave\s+Act\b", re.IGNORECASE)),
    ("ADA", re.compile(
        r"\bAmericans?\s+with\s+Disabilities\s+Act\b|\bADA\b", re.IGNORECASE)),
    ("Title VII", re.compile(
        r"\bTitle\s+VII(?:\s+of\s+the\s+Civil\s+Rights\s+Act)?\b", re.IGNORECASE)),
    ("Section 1983", re.compile(
        r"\b(?:Section|§)\s*1983\b|\b42\s+U\.?S\.?C\.?\s*§?\s*1983\b",
        re.IGNORECASE)),
    ("Lanham Act", re.compile(
        r"\bLanham\s+Act\b", re.IGNORECASE)),
    ("Sherman Act", re.compile(
        r"\bSherman\s+(?:Anti[- ]?Trust\s+)?Act\b", re.IGNORECASE)),
    ("Clayton Act", re.compile(
        r"\bClayton\s+Act\b", re.IGNORECASE)),
    ("CERCLA", re.compile(
        r"\bCERCLA\b|\bComprehensive\s+Environmental\s+Response", re.IGNORECASE)),
    ("Clean Air Act", re.compile(
        r"\bClean\s+Air\s+Act\b", re.IGNORECASE)),
    ("Clean Water Act", re.compile(
        r"\bClean\s+Water\s+Act\b", re.IGNORECASE)),
    ("NEPA", re.compile(
        r"\bNEPA\b|\bNational\s+Environmental\s+Policy\s+Act\b", re.IGNORECASE)),
    ("FOIA", re.compile(
        r"\bFOIA\b|\bFreedom\s+of\s+Information\s+Act\b", re.IGNORECASE)),
    ("Bankruptcy Code", re.compile(
        r"\bBankruptcy\s+Code\b|\b11\s+U\.?S\.?C\.?\s*§", re.IGNORECASE)),
    ("Federal Arbitration Act", re.compile(
        r"\bFederal\s+Arbitration\s+Act\b|\bFAA\b", re.IGNORECASE)),
    ("Hatch-Waxman Act", re.compile(
        r"\bHatch[- ]Waxman\s+Act\b", re.IGNORECASE)),
    ("Patent Act", re.compile(
        r"\bPatent\s+Act\b|\b35\s+U\.?S\.?C\.?\s*§", re.IGNORECASE)),
    ("Copyright Act", re.compile(
        r"\bCopyright\s+Act\b|\b17\s+U\.?S\.?C\.?\s*§", re.IGNORECASE)),
    ("Uniform Commercial Code", re.compile(
        r"\bU\.?\s*C\.?\s*C\.?\s*" + _SEC + r"\d|"
        r"\bUniform\s+Commercial\s+Code\b", re.IGNORECASE)),
    ("Class Action Fairness Act", re.compile(
        r"\bClass\s+Action\s+Fairness\s+Act\b|\bCAFA\b", re.IGNORECASE)),
    ("Federal Tort Claims Act", re.compile(
        r"\bFederal\s+Tort\s+Claims?\s+Act\b|\bFTCA\b", re.IGNORECASE)),
    ("Voting Rights Act", re.compile(
        r"\bVoting\s+Rights\s+Act\b", re.IGNORECASE)),
    ("Civil Rights Act", re.compile(
        r"\bCivil\s+Rights\s+Act(?:\s+of\s+\d{4})?\b", re.IGNORECASE)),
    ("Hobbs Act", re.compile(
        r"\bHobbs\s+Act\b", re.IGNORECASE)),
    ("Wire Fraud Statute", re.compile(
        r"\b(?:wire|mail)\s+fraud\s+statute\b|\b18\s+U\.?S\.?C\.?\s*§?\s*1343\b",
        re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Case law citation patterns  (Option B)
# ---------------------------------------------------------------------------

# Federal reporters (ordered longest-first to avoid partial matches)
_REPORTERS = (
    r"F\.\s*Supp\.\s*3d|F\.\s*Supp\.\s*2d|F\.\s*Supp\.|"
    r"F\.\s*(?:App(?:'x|x)?\.?\s*)?4th|F\.\s*3d|F\.\s*2d|"
    r"S\.\s*Ct\.|L\.\s*Ed\.\s*2d|"
    r"U\.S\.|"
    # State reporters (common)
    r"Cal\.\s*(?:App\.\s*)?(?:5th|4th|3d|2d)|Cal\.\s*Rptr\.\s*(?:3d|2d)?|"
    r"N\.Y\.\s*(?:3d|2d)|A\.D\.\s*(?:3d|2d)|"
    r"So\.\s*(?:3d|2d)|"
    r"N\.(?:E|W)\.\s*(?:3d|2d)|"
    r"A\.\s*(?:3d|2d)|"
    r"P\.\s*(?:3d|2d)|"
    r"S\.(?:E|W)\.\s*(?:2d|3d)"
)

# Full case citation: "[Volume] [Reporter] [Page]"
# Optionally preceded by a case name: "Iqbal v. Ashcroft, 556 U.S. 662 (2009)"
_CASE_CITE_RE = re.compile(
    r"\b(\d{1,4})\s+(" + _REPORTERS + r")\s+(\d{1,5})"
    r"(?:\s*,\s*(\d{1,5}))?"       # optional pinpoint page
    r"(?:\s*\([^)]{2,40}\))?"      # optional parenthetical "(S.D.N.Y. 2020)"
)

# Case name extraction: look backward from the citation for "Name v. Name"
_CASE_NAME_RE = re.compile(
    r"((?:[A-Z][A-Za-z\.\'\u2019]+(?:\s+(?:of|the|and|&|in|ex\s+rel\.?|for|de|del|van|von)\s+)?)+)"
    r"\s+v\.?\s+"
    r"((?:[A-Z][A-Za-z\.\'\u2019]+(?:\s+(?:of|the|and|&|in|for|de|del|van|von)\s+)?)+)"
    r"\s*,?\s*$"
)


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
    Scan `full_text` for all legal citations.
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
            rule_id=f"cite:{ctype}",
        )
        seen[canon] = ClauseItem(
            clause_type=ctype,
            heading=canon,
            text=ctx,
            page_start=page,
            page_end=page,
            evidence=[ev],
        )

    # ── Constitutional citations ──────────────────────────────────────────

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
        window_start = max(0, m.start() - 300)
        window_end   = min(len(full_text), m.end() + 300)
        window       = full_text[window_start:window_end]
        if not _CONST_CONTEXT_RE.search(window):
            continue
        canon = f"Article {roman}" + (f", Section {sec}" if sec else "")
        _add(canon, "article_section", m)

    # ── Federal statutes (U.S.C.) ─────────────────────────────────────────

    for m in _USC_RE.finditer(full_text):
        title = m.group(1)
        sec   = m.group(2)
        canon = f"{title} U.S.C. § {sec}"
        _add(canon, "federal_statute", m)
        sec2 = m.group(3)
        if sec2:
            canon2 = f"{title} U.S.C. § {sec2}"
            _add(canon2, "federal_statute", m)

    # ── C.F.R. ────────────────────────────────────────────────────────────

    for m in _CFR_RE.finditer(full_text):
        title = m.group(1)
        sec   = m.group(2)
        canon = f"{title} C.F.R. § {sec}"
        _add(canon, "cfr", m)

    # ── State statutes ────────────────────────────────────────────────────

    for m in _STATE_CODE_RE.finditer(full_text):
        code_name = re.sub(r"\s+", " ", m.group(1)).strip()
        sec = m.group(2)
        canon = f"{code_name} § {sec}"
        _add(canon, "state_statute", m)

    # ── Federal Rules ─────────────────────────────────────────────────────

    for rule_prefix, pat in _FEDERAL_RULES:
        for m in pat.finditer(full_text):
            num  = m.group(1)
            sub  = m.group(2) if m.lastindex >= 2 and m.group(2) else ""
            canon = f"{rule_prefix} {num}{sub}"
            _add(canon, "federal_rule", m)

    # Bare "Rule NN" — only accept in documents that already have other
    # federal rule or U.S.C. citations (avoids false positives from
    # non-legal "Rule" references)
    has_federal_context = any(
        c.clause_type in ("federal_rule", "federal_statute")
        for c in seen.values()
    )
    if has_federal_context:
        for m in _BARE_RULE_RE.finditer(full_text):
            num = m.group(1)
            sub = m.group(2) or ""
            sub = re.sub(r"\s+", "", sub)
            canon = f"Rule {num}{sub}"
            _add(canon, "federal_rule", m)

    for m in _LOCAL_RULE_RE.finditer(full_text):
        num = m.group(1)
        canon = f"Local Rule {num}"
        _add(canon, "federal_rule", m)

    # ── Named federal statutes / acts ─────────────────────────────────────

    for act_name, pat in _NAMED_ACTS:
        for m in pat.finditer(full_text):
            _add(act_name, "named_act", m)
            break  # one match per act is enough

    # ── Case law citations ────────────────────────────────────────────────

    for m in _CASE_CITE_RE.finditer(full_text):
        vol      = m.group(1)
        reporter = re.sub(r"\s+", " ", m.group(2)).strip()
        page     = m.group(3)

        # Try to extract case name from the text preceding the citation
        pre_start = max(0, m.start() - 150)
        pre_text  = full_text[pre_start:m.start()].rstrip(", \t")
        case_name = None
        name_m = _CASE_NAME_RE.search(pre_text)
        if name_m:
            p1 = name_m.group(1).strip()
            p2 = name_m.group(2).strip()
            case_name = f"{p1} v. {p2}"

        if case_name:
            canon = f"{case_name}, {vol} {reporter} {page}"
        else:
            canon = f"{vol} {reporter} {page}"

        _add(canon, "case_citation", m)

    # Sort by first appearance (char_start of first evidence)
    return sorted(seen.values(), key=lambda c: c.evidence[0].char_start)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clauses(zones: DocumentZones) -> ClauseResult:
    """
    Detect legal citations in the document.

    Returns a ClauseResult whose `clauses` list is the deduplicated set of
    legal citations (constitutional provisions, federal/state statutes,
    federal rules, named acts, case law) found anywhere in the document,
    each with a context snippet.
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
