"""
Party extraction from legal documents.

Primary: locate "v." / "vs." / "versus" separator in caption zone.
Secondary: role-keyword overrides (Plaintiff, Defendant, Petitioner, Respondent).
Merges multi-line party blocks, deduplicates, trims noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.config import (
    VERSUS_PATTERN,
    PARTY_ROLE_PATTERNS,
    PARTY_NOISE_PATTERNS,
)
from pipeline.preprocess import DocumentZones, IndexedLine, lines_to_text


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class PartyEvidence:
    source: str
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str


@dataclass
class PartyResult:
    parties: dict[str, list[str]]   # keys: plaintiffs, defendants, petitioners, respondents
    confidence: float
    evidence: list[PartyEvidence]


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_VERSUS_RE = re.compile(VERSUS_PATTERN, re.IGNORECASE)
_ROLE_COMPILED: dict[str, list[re.Pattern]] = {
    role: [re.compile(p, re.IGNORECASE) for p in pats]
    for role, pats in PARTY_ROLE_PATTERNS.items()
}
_NOISE_PATS = [re.compile(p, re.IGNORECASE) for p in PARTY_NOISE_PATTERNS]

# Lines that are unlikely to be party names.
# Covers:
#   • Court headers ("SUPERIOR COURT", "UNITED STATES DISTRICT COURT", …)
#   • Role labels with optional leading slash ("/PETITIONER:", "PLAINTIFF:", …)
#   • Navigation / procedural tokens
_SKIP_LINE = re.compile(
    r"^/?(?:"
    r"UNITED\s+STATES|U\.S\.|"
    r"IN\s+THE|FOR\s+THE|"
    r"CIVIL|CRIMINAL|"
    r"CASE\s+NO|DOCKET|"
    r"SUPERIOR\s+COURT|DISTRICT\s+COURT|CIRCUIT\s+COURT|SUPREME\s+COURT|"
    r"BANKRUPTCY\s+COURT|FAMILY\s+COURT|COURT\s+OF|STATE\s+OF|COUNTY\s+OF|"
    r"COURT|Hon\.|Judge\b|"
    r"COMPLAINT|MOTION|NOTICE|"
    r"PLAINTIFF|DEFENDANT|PETITIONER|RESPONDENT|"
    r"APPELLANT|APPELLEE|"
    r"v\.|vs\.|versus"
    r")",
    re.IGNORECASE,
)

# Also reject lines that are clearly court captions (contain "[PROPOSED]",
# "ORDER", or "MEMORANDUM" alongside court language)
_COURT_CAPTION_IN_PARTY = re.compile(
    r"""
    \[PROPOSED\]
    | \bORDER\b.*\bCOURT\b | \bCOURT\b.*\bORDER\b
    | ^CAUSE\s+NO\.                          # Texas state court docket line
    | ^FORM\s+NO\.                           # Texas citation form header
    | \b(?:HOUSTON|DALLAS|SAN\s+ANTONIO|AUSTIN|EL\s+PASO|FORT\s+WORTH)\s+DIVISION\b
                                             # Texas federal/state district divisions
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Sentence verbs that indicate the "name" is really a body-text fragment.
# e.g. "Plaintiffs will file a Second Amended Complaint" or
#      "Defendants move to dismiss" — these appear in scheduling orders
#      where Plaintiff/Defendant are used as references, not party labels.
_PARTY_SENTENCE_VERB = re.compile(
    r"\b(?:will|shall|may|have|has|had|is|are|was|were|"
    r"move[sd]?|file[sd]?|alleg(?:e[sd]?|ing)|seek[s]?|sought|"
    r"represent[s]?|answer[s]?|dismiss(?:ed)?|respond[s]?|"
    r"oppos(?:e[sd]?|ing)|purchas(?:e[sd]?|ing)|brought)\b",
    re.IGNORECASE,
)

_MAX_PARTY_NAME_LEN = 350   # multi-defendant blocks can be long
_MIN_PARTY_NAME_LEN = 2

# Single-word entries that can never be party names
_BARE_CONJUNCTION = frozenset({"AND", "OR", "THE", "A", "AN", "IN", "OF", "FOR", "TO", "BY"})

# Signals that a string is an attorney signature block / contact info, not a party name
_ATTORNEY_BLOCK_RE = re.compile(
    r"""
    \b\d{3}[/\-\.]\d{3,4}[/\-\.]\d{4}\b   # phone: 310/201-9150 or 310-201-9150
    | \(\d{3}\)\s*\d{3}[\-\s]\d{4}          # phone: (310) 201-9150
    | \(fax\)                                 # fax label
    | \w[\w.+-]+@\w[\w.-]+\.\w{2,}           # email address
    | \bSuite\s+\d+\b                         # Suite 2100
    | \bFloor\s+\d+\b                         # Floor 12
    | \b\d+\s+[A-Z][a-z]+\s+(?:Street|Avenue|Boulevard|Drive|Road|Park|Way|Lane|Blvd|Ave|St\.)\b
    | \(\d{5,6}\)                             # CA bar number: (274241)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Parenthetical role labels embedded in body prose: Oscar Gonzalez ("Plaintiff")
_PAREN_ROLE_RE = re.compile(
    r'\s*[\(\[][\u201c\u201d"\']*(?:Plaintiff|Defendant|Petitioner|Respondent)s?[\u201c\u201d"\']*[\)\]]',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_party_name(raw: str) -> str:
    """Strip noise tokens and normalize whitespace."""
    text = raw.strip()
    # Strip leading role-label slash artifacts: "/PETITIONER:", "/RESPONDENT:"
    text = re.sub(r"^/\s*(?:PETITIONER|RESPONDENT|PLAINTIFF|DEFENDANT|APPELLANT|APPELLEE)S?\s*:?\s*",
                  "", text, flags=re.IGNORECASE)
    # Strip parenthetical role labels: John Smith ("Plaintiff") → John Smith
    text = _PAREN_ROLE_RE.sub("", text)
    for pat in _NOISE_PATS:
        text = pat.sub("", text)
    # Remove trailing punctuation artifacts: comma, semicolon, period, colon
    text = re.sub(r"[,;\.:\-]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid_party(name: str) -> bool:
    name = name.strip()
    if len(name) < _MIN_PARTY_NAME_LEN or len(name) > _MAX_PARTY_NAME_LEN:
        return False
    if not re.search(r"[A-Za-z]", name):
        return False
    if _COURT_CAPTION_IN_PARTY.search(name):
        return False
    # Must start with a capital letter — filters "and lead counsel in", "allege that…"
    if not name[0].isupper():
        return False
    # Reject sentence fragments: body text where Plaintiff/Defendant is a reference
    if _PARTY_SENTENCE_VERB.search(name):
        return False
    # Reject bare conjunctions / articles standing alone (e.g. "AND", "OR", "THE")
    if name.upper().strip() in _BARE_CONJUNCTION:
        return False
    # Reject attorney signature blocks (phone, email, bar number, address)
    if _ATTORNEY_BLOCK_RE.search(name):
        return False
    # Reject camelCase merge artifacts — a word where a lowercase letter is
    # immediately followed by an uppercase letter (e.g. "BrandJOHN") indicates
    # two tokens were fused at a line-break during PDF/DOCX extraction.
    for word in name.split():
        if re.search(r'[a-z][A-Z]', word):
            return False
    return True


def _merge_multiline_block(lines: list[IndexedLine], max_gap: int = 2) -> list[str]:
    """
    Merge consecutive non-empty lines into party name blocks.
    A blank line or gap > max_gap line numbers ends a block.
    Returns list of merged party strings.
    """
    if not lines:
        return []

    blocks: list[str] = []
    current: list[str] = []
    prev_line_num = -999

    for il in lines:
        text = il.text.strip()
        if not text:
            if current:
                blocks.append(" ".join(current))
                current = []
            prev_line_num = -999
            continue

        gap = il.line_num - prev_line_num
        if gap > max_gap and current:
            blocks.append(" ".join(current))
            current = []

        current.append(text)
        prev_line_num = il.line_num

    if current:
        blocks.append(" ".join(current))

    return blocks


_PLAINTIFF_LABEL = re.compile(r"^\s*plaintiff", re.IGNORECASE)
_DEFENDANT_LABEL = re.compile(r"^\s*defendant", re.IGNORECASE)

# Court location lines that appear inside the caption table but are not parties.
_COURT_LOCATION_LINE = re.compile(
    r"(?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)\s+DISTRICT\b|"
    r"\bDISTRICT\s+OF\s+[A-Z]|"
    r"\bCOUNTY\s+OF\s+[A-Z]",
    re.IGNORECASE,
)


def _find_versus_split(
    caption_lines: list[IndexedLine],
) -> tuple[list[IndexedLine], list[IndexedLine], Optional[int]]:
    """
    Find the "v." separator in the caption zone and return (above, below, vs_line_num).

    Uses role labels (Plaintiff / Defendants) as explicit block boundaries so that
    consecutive lines in a DOCX table cell (no blank lines between court header,
    party name, and role label) are split correctly.

    Logic:
      • Plaintiffs:  lines between the last structural court-header line and the
                     "Plaintiff[s]," role label directly above "v."
      • Defendants:  lines between "v." and the "Defendant[s]." role label below it
    """
    for v_idx, il in enumerate(caption_lines):
        if not _VERSUS_RE.search(il.text):
            continue

        above_all = caption_lines[:v_idx]
        below_all = caption_lines[v_idx + 1:]

        # ── Plaintiffs: find the role label just above "v." ────────────────
        plaintiff_label_idx: Optional[int] = None
        for j in range(len(above_all) - 1, -1, -1):
            if _PLAINTIFF_LABEL.match(above_all[j].text):
                plaintiff_label_idx = j
                break

        if plaintiff_label_idx is not None:
            # Court header: last SKIP_LINE (non-plaintiff) before the role label
            last_header_idx = -1
            for j in range(plaintiff_label_idx - 1, -1, -1):
                txt = above_all[j].text.strip()
                if _SKIP_LINE.match(txt) and not _PLAINTIFF_LABEL.match(txt):
                    last_header_idx = j
                    break
            # Party name lines: between court header and role label,
            # excluding any lingering court-location lines
            above_lines = [
                l for l in above_all[last_header_idx + 1 : plaintiff_label_idx]
                if not _COURT_LOCATION_LINE.search(l.text)
            ]
        else:
            # No explicit role label; filter structural lines and hope for the best
            above_lines = [
                l for l in above_all
                if not _SKIP_LINE.match(l.text.strip())
                and not _COURT_LOCATION_LINE.search(l.text)
            ]

        # ── Defendants: lines from "v." until the role label ──────────────
        below_lines: list[IndexedLine] = []
        for l in below_all:
            txt = l.text.strip()
            if _DEFENDANT_LABEL.match(txt) or _SKIP_LINE.match(txt):
                break
            below_lines.append(l)

        return above_lines, below_lines, il.line_num

    return [], [], None


def _extract_role_overrides(
    text: str,
    parties: dict[str, list[str]],
    evidence: list[PartyEvidence],
) -> None:
    """
    Look for explicit role labels (Plaintiff:, Defendant:, etc.) and extract
    the associated name from the same line or next line.
    Updates parties dict in-place.

    Only fires for roles not already populated by versus_split, and only when
    the role keyword appears at the START of a line — this prevents matching
    "LEAD PLAINTIFF AND APPROVAL OF" embedded mid-sentence in a title.
    """
    for role, pats in _ROLE_COMPILED.items():
        canonical_role = _canonical_role(role)
        if canonical_role is None:
            continue
        # Skip this role if versus_split already found parties for it
        if parties.get(canonical_role):
            continue
        for pat in pats:
            for m in pat.finditer(text):
                # Require the match to be at the start of a line (or document)
                pos = m.start()
                if pos > 0 and text[pos - 1] != "\n":
                    continue
                # Take text after the keyword on the same line
                rest = text[m.end():].split("\n")[0].strip().lstrip(":,; ")
                if rest and _is_valid_party(rest):
                    name = _clean_party_name(rest.split(",")[0])
                    if name and name not in parties[canonical_role]:
                        parties[canonical_role].append(name)
                        evidence.append(PartyEvidence(
                            source="rule",
                            page=1,
                            span_text=m.group(0) + " " + name,
                            char_start=m.start(),
                            char_end=m.end() + len(name),
                            rule_id=f"party_role:{role}",
                        ))


def _canonical_role(role: str) -> Optional[str]:
    mapping = {
        "plaintiffs":  "plaintiffs",
        "defendants":  "defendants",
        "petitioners": "petitioners",
        "respondents": "respondents",
        "appellants":  "plaintiffs",   # appellants treated as plaintiffs
        "appellees":   "defendants",   # appellees treated as defendants
    }
    return mapping.get(role)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_parties(zones: DocumentZones) -> PartyResult:
    """
    Extract plaintiffs, defendants, petitioners, and respondents.

    Strategy:
    1. Locate "v." in caption zone → text above = plaintiffs, below = defendants.
    2. Apply role-keyword overrides.
    3. Clean, merge, deduplicate.
    """
    parties: dict[str, list[str]] = {
        "plaintiffs":  [],
        "defendants":  [],
        "petitioners": [],
        "respondents": [],
    }
    evidence: list[PartyEvidence] = []

    caption_lines = zones.caption_zone
    caption_text  = lines_to_text(caption_lines)

    above_lines, below_lines, versus_line_num = _find_versus_split(caption_lines)

    if versus_line_num is not None:
        # Merge multi-line blocks above and below the "v." line
        above_blocks = _merge_multiline_block(above_lines)
        below_blocks  = _merge_multiline_block(below_lines)

        for raw in above_blocks:
            name = _clean_party_name(raw)
            if _is_valid_party(name) and not _SKIP_LINE.match(name):
                parties["plaintiffs"].append(name)

        for raw in below_blocks:
            name = _clean_party_name(raw)
            if _is_valid_party(name) and not _SKIP_LINE.match(name):
                parties["defendants"].append(name)

        if parties["plaintiffs"] or parties["defendants"]:
            ev_text = caption_text[
                max(0, caption_text.rfind(above_blocks[-1]) if above_blocks else 0):
                caption_text.find(below_blocks[0]) + len(below_blocks[0]) if below_blocks else 200
            ]
            evidence.append(PartyEvidence(
                source="rule",
                page=caption_lines[0].page_num if caption_lines else 1,
                span_text=ev_text[:200],
                char_start=0,
                char_end=min(200, len(caption_text)),
                rule_id="versus_split",
            ))

    # Role-keyword overrides applied to entire first-page text
    first_page_text = lines_to_text(zones.first_page_zone)
    _extract_role_overrides(first_page_text, parties, evidence)

    # Deduplicate
    for role in parties:
        seen: set[str] = set()
        deduped: list[str] = []
        for name in parties[role]:
            key = name.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(name)
        parties[role] = deduped

    found_any = any(parties.values())
    confidence = 0.85 if versus_line_num is not None else (0.5 if found_any else 0.0)

    return PartyResult(parties=parties, confidence=confidence, evidence=evidence)
