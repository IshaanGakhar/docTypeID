"""
Title extraction using regex candidate selection + TF-IDF cosine-similarity ranking.

Compound titles are handled: after selecting the top candidate line, adjacent
continuation lines (starting with AND, FOR, OF, IN, …) are merged to form the
complete title string.

Examples:
    "MOTION TO DISMISS"          → compound with next line →
    "AND CHANGE OF VENUE"
    becomes → "MOTION TO DISMISS AND CHANGE OF VENUE"

    "NOTICE OF APPOINTMENT OF COUNSEL"
    "AND SETTLEMENT"
    becomes → "NOTICE OF APPOINTMENT OF COUNSEL AND SETTLEMENT"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.config import (
    TITLE_PREFIXES,
    TITLE_BOOST_PHRASES,
    OPENING_PATTERNS,
    TITLE_MIN_CAPS_LEN,
    TFIDF_TITLE_BOOST,
    TITLE_CONTINUATION_WORDS,
    TITLE_CONTINUATION_MAX_GAP,
)
from pipeline.preprocess import DocumentZones, IndexedLine, lines_to_text


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class TitleEvidence:
    source: str           # "regex" | "tfidf"
    page: int
    span_text: str
    char_start: int
    char_end: int
    rule_id: str
    score: float = 0.0


@dataclass
class TitleResult:
    title: Optional[str]
    confidence: float
    evidence: list[TitleEvidence]


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Include curly/smart quotes (U+2018/2019/201C/201D) — PDFs frequently embed
# these instead of straight apostrophes/quotes, e.g. "STOLLE\u2019S".
_CAPS_LINE = re.compile(
    r"^[A-Z][A-Z0-9\s\(\)\-\:\,\.\'\"\u2018\u2019\u201c\u201d]{%d,}$" % (TITLE_MIN_CAPS_LEN - 1)
)
_PREFIX_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in TITLE_PREFIXES) + r")\b",
    re.IGNORECASE,
)
_OPENING_PATTERN = re.compile(
    "|".join(OPENING_PATTERNS), re.IGNORECASE | re.MULTILINE
)
_BOOST_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TITLE_BOOST_PHRASES]

# Lines that identify a court header — must NOT be treated as title candidates.
# These are the first thing in every legal document and always score highly on
# TF-IDF, causing court names to be returned as the document title.
_COURT_HEADER_RE = re.compile(
    r"^(?:"
    # Court names / jurisdiction headers
    r"UNITED\s+STATES|"
    r"IN\s+THE\s+(?:UNITED|DISTRICT|SUPERIOR|CIRCUIT|SUPREME|COURT|STATE)|"
    r"SUPERIOR\s+COURT|"
    r"DISTRICT\s+COURT|"
    r"CIRCUIT\s+COURT|"
    r"SUPREME\s+COURT|"
    r"COURT\s+OF\s+APPEALS|"
    r"BANKRUPTCY\s+COURT|"
    r"FAMILY\s+COURT|"
    r"(?:THE\s+)?STATE\s+OF\s+[A-Z]|"
    r"COUNTY\s+OF\s+[A-Z]|"
    r"FOR\s+(?:THE\s+)?(?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)\s+DISTRICT|"
    r"(?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)\s+DISTRICT|"
    # Location fragments that bleed into title zone ("DALLAS COUNTY", "HARRIS COUNTY")
    r"[A-Z][A-Z\s]+\s+COUNTY\b|"
    # Docket / cause number headers (Texas state court and federal)
    r"CAUSE\s+NO\b|"
    r"(?:LEAD\s+)?CASE\s+NO\b|"
    r"CIVIL\s+ACTION\s+NO\b|"
    # Case name captions — never a document title
    r"IN\s+RE\b|"
    # Common case-name suffixes appearing alone after "IN RE" is split off
    r"SECURITIES\s+LITIGATION|"
    r"SHAREHOLDER\s+LITIGATION|"
    r"CLASS\s+ACTION|"
    # Consolidation phrases used as sub-headings in multi-case filings
    r"ALL\s+ACTIONS|"
        # Section headings that score highly but are structural, not titles
    r"TABLE\s+OF\s+(?:CONTENTS|AUTHORITIES)|"
    r"(?:ARGUMENT|CONCLUSION|INTRODUCTION|DISCUSSION|ANALYSIS|BACKGROUND|SUMMARY|"
    r"STATEMENT\s+OF|LEGAL\s+STANDARD|STANDARD\s+OF\s+REVIEW|PRAYER\s+FOR\s+RELIEF|"
    r"RELIEF\s+REQUESTED)\s*$|"
    # Texas / other state court filing stamps
    r"DISTRICT\s+CLERK|"
    # Judicial closing lines
    r"IT\s+IS\s+(?:HEREBY\s+)?(?:SO\s+)?ORDER"
    r")",
    re.IGNORECASE,
)

# Continuation line: starts with one of the configured continuation words
_CONTINUATION_RE = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in TITLE_CONTINUATION_WORDS) + r")\b",
    re.IGNORECASE,
)

# A line that is entirely a short ALL-CAPS fragment also qualifies as continuation
_SHORT_CAPS_FRAGMENT = re.compile(r"^[A-Z][A-Z\s\-]{1,40}$")


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _is_caps_line(text: str) -> bool:
    return bool(_CAPS_LINE.match(text.strip()))


_FILING_STAMP_RE = re.compile(
    r"DISTRICT\s+CLERK|CIRCUIT\s+CLERK|COUNTY\s+CLERK|CLERK\s+OF\s+COURT",
    re.IGNORECASE,
)
_FILED_LINE_RE = re.compile(r"^E?-?FILED\s*$", re.IGNORECASE)


def _build_stamp_line_nums(lines: list[IndexedLine]) -> set[int]:
    """
    Return line numbers that belong to a court filing stamp block.

    The Texas filing stamp layout is:
        FILED           ← lone keyword line
        [COUNTY] COUNTY
        [DATE/TIME]
        [CLERK NAME]    ← what we must exclude from title candidates
        DISTRICT CLERK  ← anchor

    Strategy: only look back from DISTRICT/CIRCUIT/COUNTY CLERK lines (the
    anchor), NOT from bare FILED lines — those are too far from the actual
    clerk name and would wrongly mark real document content as stamp lines.
    We look back at most 2 lines from the CLERK anchor.
    """
    stamp_nums: set[int] = set()
    for idx, il in enumerate(lines):
        if _FILED_LINE_RE.match(il.text.strip()):
            # Mark only the FILED line itself; don't look back
            stamp_nums.add(il.line_num)
        elif _FILING_STAMP_RE.search(il.text):
            stamp_nums.add(il.line_num)
            # Mark the 2 lines immediately before the CLERK anchor
            # (clerk's name + timestamp — never actual document heading)
            for prev in lines[max(0, idx - 2): idx]:
                stamp_nums.add(prev.line_num)
    return stamp_nums


def _collect_candidates(lines: list[IndexedLine]) -> list[tuple[IndexedLine, str]]:
    """
    Returns list of (IndexedLine, reason) for candidate title lines.
    """
    candidates: list[tuple[IndexedLine, str]] = []
    seen_texts: set[str] = set()

    stamp_line_nums = _build_stamp_line_nums(lines)

    for il in lines:
        text = il.text.strip()
        if not text or len(text) < 5 or len(text) > 250:
            continue
        if text in seen_texts:
            continue

        # Never use a court-header line as a title candidate
        if _COURT_HEADER_RE.match(text):
            continue

        # Skip lines that are part of a filing stamp (e.g. clerk's name)
        if il.line_num in stamp_line_nums:
            continue

        # Skip form-field labels: lines that end with a bare colon ("STREET ADDRESS:")
        if text.endswith(":") and len(text.split()) <= 4:
            continue

        # Skip table-of-contents entries: lines with dotted leaders (e.g.
        # "INTRODUCTION .......................................................1")
        if text.count(".") >= 4 and re.search(r"\.{3,}", text):
            continue

        if _is_caps_line(text) and len(text) >= TITLE_MIN_CAPS_LEN:
            # Reject bare role labels: "PLAINTIFF(S)," / "DEFENDANT," etc.
            role_stripped = re.sub(
                r"^(?:PLAINTIFF|DEFENDANT|PETITIONER|RESPONDENT)S?\s*[\(\)S]*\s*[,\.]?\s*$",
                "", text, flags=re.IGNORECASE,
            ).strip()
            if not role_stripped:
                continue
            candidates.append((il, "caps_line"))
            seen_texts.add(text)
            continue

        if _PREFIX_PATTERN.match(text):
            candidates.append((il, "title_prefix"))
            seen_texts.add(text)
            continue

        if _OPENING_PATTERN.match(text):
            # Reject bare role labels like "Plaintiff," or "Defendant." with nothing
            # substantive after the keyword — these are party labels, not titles.
            stripped = re.sub(r"^(?:plaintiff|defendant|petitioner|respondent)s?\b\s*[,\.]?\s*",
                              "", text, flags=re.IGNORECASE).strip()
            if not stripped:
                continue
            candidates.append((il, "opening_pattern"))
            seen_texts.add(text)

    return candidates


# ---------------------------------------------------------------------------
# Compound title merging
# ---------------------------------------------------------------------------

def _is_continuation_line(line_text: str) -> bool:
    """
    Return True if a line looks like it continues the previous title line.
    Matches lines starting with AND/FOR/OF/IN/… or short ALL-CAPS fragments.
    """
    t = line_text.strip()
    if not t:
        return False
    if _CONTINUATION_RE.match(t):
        return True
    # Short ALL-CAPS fragment that could be "AND CHANGE OF VENUE" remainder
    if _SHORT_CAPS_FRAGMENT.match(t) and len(t) <= 50:
        return True
    return False


def _merge_compound_title(
    top_il: IndexedLine,
    all_lines: list[IndexedLine],
    candidate_line_nums: set[int],
) -> tuple[str, int]:
    """
    Starting from top_il, scan forward for continuation lines and merge them.

    A continuation line must:
    - Follow within TITLE_CONTINUATION_MAX_GAP line numbers
    - Match _is_continuation_line()
    - Not itself be a standalone title candidate (not in candidate_line_nums),
      OR be a continuation word start like "AND …"

    Returns (merged_title_text, last_char_end).
    """
    parts = [top_il.text.strip()]
    last_char_end = top_il.char_start + len(top_il.text)
    prev_line_num = top_il.line_num

    # Find this line's position in all_lines
    start_idx = next(
        (i for i, il in enumerate(all_lines) if il.line_num == top_il.line_num),
        -1,
    )
    if start_idx == -1:
        return top_il.text.strip(), last_char_end

    for il in all_lines[start_idx + 1:]:
        gap = il.line_num - prev_line_num
        if gap > TITLE_CONTINUATION_MAX_GAP:
            break

        text = il.text.strip()
        if not text:
            continue

        # Never absorb a court-header line into the title, even if it looks
        # like a short ALL-CAPS continuation fragment
        if _COURT_HEADER_RE.match(text):
            break

        if not _is_continuation_line(text):
            break

        # Stop if the line is clearly sentence-case prose (starts lowercase
        # after being stripped, or has many lower-case words) — this catches
        # cases where a title ends and body text begins on the next line
        # even if that line starts with a continuation word like "Pursuant".
        words = text.split()
        lower_words = sum(1 for w in words if w and w[0].islower())
        if lower_words > 2:
            break

        # Hard cap: merged titles longer than 200 chars are almost certainly
        # running into body text.
        if len(" ".join(parts) + " " + text) > 200:
            break

        # Don't merge a line that is its own strong standalone candidate
        # (e.g., a completely different heading right after) — UNLESS the
        # accumulated title so far ends with a hanging continuation word like
        # "AND" or "OF", which means the title was split mid-phrase across lines.
        accumulated_so_far = " ".join(parts)
        trailing_word = accumulated_so_far.rstrip().rsplit(None, 1)[-1].upper().rstrip(",;")
        hangs = trailing_word in {w.upper() for w in TITLE_CONTINUATION_WORDS}
        if il.line_num in candidate_line_nums and not _CONTINUATION_RE.match(text) and not hangs:
            break

        parts.append(text)
        last_char_end = il.char_start + len(text)
        prev_line_num = il.line_num

    merged = " ".join(parts)
    return merged, last_char_end


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_boost(text: str) -> float:
    for pat in _BOOST_PATTERNS:
        if pat.search(text):
            return TFIDF_TITLE_BOOST
    return 1.0


def _rank_candidates(
    candidates: list[tuple[IndexedLine, str]],
    context_text: str,
) -> list[tuple[IndexedLine, str, float]]:
    if not candidates:
        return []

    texts = [il.text for il, _ in candidates] + [context_text]

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="word",
                              sublinear_tf=True)
        tfidf_mat = vec.fit_transform(texts)
    except ValueError:
        return [(il, reason, 0.0) for il, reason in candidates]

    context_vec = tfidf_mat[-1]
    candidate_vecs = tfidf_mat[:-1]
    sims = cosine_similarity(candidate_vecs, context_vec).flatten()

    scored: list[tuple[IndexedLine, str, float]] = []
    for i, (il, reason) in enumerate(candidates):
        boost = _compute_boost(il.text)
        base  = sims[i]
        if reason == "opening_pattern":
            base *= 0.5
        elif reason == "title_prefix":
            # title_prefix lines start with a legal doc-type keyword (MOTION, ORDER, …)
            # and are the most reliable signal — boost them above generic caps_line matches
            # like "SECURITIES LITIGATION" or "IN RE [CASE NAME]" fragments
            base *= 1.4
        scored.append((il, reason, base * boost))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_title(zones: DocumentZones) -> TitleResult:
    """
    Extract document title from DocumentZones.

    Strategy:
    1. Collect candidates from title_zone (first ~60 lines).
    2. Rank via TF-IDF cosine similarity against opening paragraph.
    3. Merge adjacent continuation lines into a compound title.
       e.g. "MOTION TO DISMISS" + "AND CHANGE OF VENUE"
            → "MOTION TO DISMISS AND CHANGE OF VENUE"
    4. Return merged title with evidence and confidence.
    """
    title_lines  = zones.title_zone
    all_lines    = zones.all_lines
    context_text = lines_to_text(zones.body_zone[:20])

    candidates = _collect_candidates(title_lines)

    if not candidates:
        return TitleResult(title=None, confidence=0.0, evidence=[])

    ranked = _rank_candidates(candidates, context_text)

    if not ranked:
        return TitleResult(title=None, confidence=0.0, evidence=[])

    top_il, top_reason, top_score = ranked[0]

    # Set of line_nums that are standalone candidates (used to guard merging)
    candidate_line_nums = {il.line_num for il, _ in candidates}

    # Merge compound title from continuation lines
    title_text, char_end = _merge_compound_title(top_il, all_lines, candidate_line_nums)

    confidence = float(min(1.0, top_score + 0.1))

    evidence = [
        TitleEvidence(
            source="regex" if top_reason in ("caps_line", "title_prefix", "opening_pattern") else "tfidf",
            page=top_il.page_num,
            span_text=title_text,
            char_start=top_il.char_start,
            char_end=char_end,
            rule_id=top_reason + (":compound" if " " in title_text[len(top_il.text.strip()):] else ""),
            score=top_score,
        )
    ]

    return TitleResult(title=title_text, confidence=confidence, evidence=evidence)
