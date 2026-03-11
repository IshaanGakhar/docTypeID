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
    CITATION_BOILERPLATE_RE_STR,
    DOCUMENT_TYPE_LABELS,
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
    # Bare "DISTRICT OF <state>" lines (e.g. "DISTRICT OF MARYLAND")
    r"DISTRICT\s+OF\s+[A-Z][A-Za-z]+\s*$|"
    # "FOR THE DISTRICT OF ..."
    r"FOR\s+THE\s+DISTRICT\s+OF\b|"
    # Location fragments that bleed into title zone ("DALLAS COUNTY", "HARRIS COUNTY")
    # Limited to 1-3 words before COUNTY to avoid matching long title lines
    # like "TRANSFER VENUE TO MONTGOMERY COUNTY"
    r"[A-Z]{2,20}(?:\s+[A-Z]{2,15}){0,2}\s+COUNTY\s*(?:,|$)|"
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
# Allow periods for middle initials (e.g. "RANDI W. SINGER AS")
_SHORT_CAPS_FRAGMENT = re.compile(r"^[A-Z][A-Z\s\.\-]{1,40}$")

# Citation boilerplate: lines containing this text are instructional boilerplate
# from Texas state citation forms — never a valid title.
_CITATION_BP_RE = re.compile(CITATION_BOILERPLATE_RE_STR, re.IGNORECASE)
_CITATION_NOISE_RE = re.compile(
    r"(?:answer\s+with\s+the\s+clerk|"
    r"employ\s+an\s+attorney|"
    r"default\s+judgment\s+may\s+be\s+taken|"
    r"petition\s+was\s+filed|"
    r"officer\s+executing\s+this\s+(?:citation|writ))",
    re.IGNORECASE,
)


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


_PARTY_ROLE_LINE = re.compile(
    r"^\s*(?:plaintiff|defendant|petitioner|respondent|appellee|appellant)s?"
    r"\s*[,.\s/]*$",
    re.IGNORECASE,
)
_VS_LINE = re.compile(r"^\s*(?:v\.|vs\.?|versus)\s*$", re.IGNORECASE)

_PARTY_BLOCK_STOP = re.compile(
    r"(?:"
    r"CASE\s+NO|CAUSE\s+NO|CIVIL\s+ACTION|CLASS\s+ACTION|"
    r"PROOF\s+OF|MOTION|NOTICE|ORDER|MEMORANDUM|COMPLAINT|"
    r"PETITION|DECLARATION|AFFIDAVIT|STIPULATION|SUBPOENA|"
    r"BRIEF|REPLY|RESPONSE|OPPOSITION|SUMMONS|REPORT|"
    r"\[caption\s+continued"
    r")",
    re.IGNORECASE,
)


def _build_party_block_nums(lines: list[IndexedLine]) -> set[int]:
    """Return line numbers that sit inside the caption party block.

    Detects the region between role labels (e.g. "Plaintiff," / "v." /
    "Defendants.") and marks every line in between — these are party names
    and must never be treated as title candidates.

    After "v.", scans forward until a closing role label, a doc-type keyword,
    or end of zone — whichever comes first.  Before "v.", scans backward to
    the nearest role label or court header.
    """
    party_nums: set[int] = set()
    vs_idx: int | None = None
    role_indices: list[int] = []

    for idx, il in enumerate(lines):
        t = il.text.strip()
        if _VS_LINE.match(t):
            vs_idx = idx
            party_nums.add(il.line_num)
        elif _PARTY_ROLE_LINE.match(t):
            role_indices.append(idx)
            party_nums.add(il.line_num)

    if vs_idx is None:
        return party_nums

    # Backward from "v.": include lines up to the nearest role label before it,
    # but skip lines that contain doc-type keywords (in two-column captions the
    # title can appear between "Plaintiff," and "v.")
    pre_roles = [r for r in role_indices if r < vs_idx]
    block_start = pre_roles[-1] if pre_roles else vs_idx
    for idx in range(block_start, vs_idx):
        t = lines[idx].text.strip()
        if _PARTY_BLOCK_STOP.search(t):
            continue
        party_nums.add(lines[idx].line_num)

    # Forward from "v.": include lines until a role label, doc-type keyword,
    # or a bare ")" line with nothing else (bracket separator end)
    post_roles = [r for r in role_indices if r > vs_idx]
    block_end_limit = post_roles[0] if post_roles else len(lines) - 1

    for idx in range(vs_idx + 1, min(block_end_limit + 1, len(lines))):
        t = lines[idx].text.strip()
        if not t or t == ")":
            party_nums.add(lines[idx].line_num)
            continue
        if _PARTY_BLOCK_STOP.search(t):
            break
        if _COURT_HEADER_RE.match(t):
            break
        party_nums.add(lines[idx].line_num)

    return party_nums


def _collect_candidates(lines: list[IndexedLine]) -> list[tuple[IndexedLine, str]]:
    """
    Returns list of (IndexedLine, reason) for candidate title lines.
    """
    candidates: list[tuple[IndexedLine, str]] = []
    seen_texts: set[str] = set()

    stamp_line_nums = _build_stamp_line_nums(lines)
    party_block_nums = _build_party_block_nums(lines)

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

        # Skip lines inside the caption party block (between role labels)
        if il.line_num in party_block_nums:
            continue

        # Skip form-field labels: lines that end with a bare colon ("STREET ADDRESS:")
        if text.endswith(":") and len(text.split()) <= 4:
            continue

        # Skip citation boilerplate lines (Texas service forms)
        if _CITATION_BP_RE.search(text) or _CITATION_NOISE_RE.search(text):
            continue

        # Skip table-of-contents entries: lines with dotted leaders (e.g.
        # "INTRODUCTION .......................................................1")
        if text.count(".") >= 4 and re.search(r"\.{3,}", text):
            continue

        if _is_caps_line(text) and len(text) >= TITLE_MIN_CAPS_LEN:
            role_stripped = re.sub(
                r"^(?:PLAINTIFF|DEFENDANT|PETITIONER|RESPONDENT)S?\s*[\(\)S]*\s*[,\.]?\s*$",
                "", text, flags=re.IGNORECASE,
            ).strip()
            if not role_stripped:
                continue
            # Skip phone/fax lines ("T: 212-363-7500", "F: 212-363-7171")
            if re.match(r"^[A-Z]:\s*[\d\-\(\)\s]+$", text):
                continue
            # Skip single-word ALL-CAPS fragments that are likely party name
            # fragments or noise (e.g. "MATERIALS", "INTERNATIONAL") — but
            # keep words that match a known document type label.
            word_count = len(text.rstrip(",.;:").split())
            if word_count == 1 and not _PREFIX_PATTERN.match(text):
                clean_word = text.strip().rstrip(",.;:").upper()
                if clean_word not in {l.upper() for l in DOCUMENT_TYPE_LABELS}:
                    continue
            candidates.append((il, "caps_line"))
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
    _HANG_WORDS = {w.upper() for w in TITLE_CONTINUATION_WORDS}
    _HANG_WORDS |= {"DEFENDANT", "DEFENDANTS", "PLAINTIFF", "PLAINTIFFS",
                    "PETITIONER", "PETITIONERS", "RESPONDENT", "RESPONDENTS"}

    # Find this line's position in all_lines
    start_idx = next(
        (i for i, il in enumerate(all_lines) if il.line_num == top_il.line_num),
        -1,
    )
    if start_idx == -1:
        return top_il.text.strip(), top_il.char_start + len(top_il.text)

    # Backward look: if the preceding line is a candidate AND ends with a
    # hanging word (possessive, preposition, role keyword), start the merge
    # from that earlier line instead.  This handles split titles where TF-IDF
    # ranked the second half higher than the first.
    actual_start = start_idx
    for back_i in range(start_idx - 1, max(start_idx - 3, -1), -1):
        prev_il = all_lines[back_i]
        if prev_il.line_num not in candidate_line_nums:
            break
        gap = all_lines[back_i + 1].line_num - prev_il.line_num
        if gap > TITLE_CONTINUATION_MAX_GAP:
            break
        prev_text = prev_il.text.strip()
        if not prev_text:
            continue
        if _COURT_HEADER_RE.match(prev_text):
            break
        trailing = re.sub(r"['\u2019,;]+$", "", prev_text.rsplit(None, 1)[-1]).upper()
        if trailing in _HANG_WORDS:
            actual_start = back_i
        else:
            break

    if actual_start < start_idx:
        top_il = all_lines[actual_start]
        start_idx = actual_start

    parts = [top_il.text.strip()]
    last_char_end = top_il.char_start + len(top_il.text)
    prev_line_num = top_il.line_num

    for il in all_lines[start_idx + 1:]:
        gap = il.line_num - prev_line_num
        if gap > TITLE_CONTINUATION_MAX_GAP:
            break

        text = il.text.strip()
        if not text:
            continue

        # Never absorb a court-header line, filing stamp, or versus line
        if _COURT_HEADER_RE.match(text):
            break
        if _FILED_LINE_RE.match(text):
            break
        if _VS_LINE.match(text):
            break

        # Stop at salutation / address lines that follow legal titles
        if re.match(r"^TO\s+THE\s+(?:CLERK|COURT|HONOR)", text, re.IGNORECASE):
            break
        if re.match(r"^(?:PLEASE\s+TAKE\s+NOTICE|COMES?\s+NOW|NOW\s+COMES?)\b", text, re.IGNORECASE):
            break

        # Check if the accumulated title so far ends with a hanging word
        # (preposition, conjunction, or role keyword like DEFENDANT) that
        # grammatically demands the next line to complete the phrase.
        accumulated_so_far = " ".join(parts)
        trailing_raw = accumulated_so_far.rstrip().rsplit(None, 1)[-1].upper()
        trailing_word = re.sub(r"['\u2019,;]+$", "", trailing_raw)
        hangs = trailing_word in _HANG_WORDS

        is_cont = _is_continuation_line(text)

        # If the line doesn't look like a continuation AND the previous
        # title doesn't hang, stop merging.
        if not is_cont and not hangs:
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
        # accumulated title hangs or the line starts with a continuation word.
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
            # Opening patterns that contain a doc-type keyword (PETITION,
            # COMPLAINT, MOTION …) are legitimate titles and should not be
            # penalized.  Only penalize those that are purely procedural
            # (e.g. "Comes Now" boilerplate without a doc-type keyword).
            if _PREFIX_PATTERN.match(il.text.strip()):
                base *= 1.2
            else:
                base *= 0.5
        elif reason == "title_prefix":
            base *= 1.4
        scored.append((il, reason, base * boost))

    scored.sort(key=lambda x: x[2], reverse=True)

    # Fallback: when all TF-IDF scores are 0.0 (e.g. empty body text on
    # short documents), use heuristic scoring based on whether the line
    # starts with a known doc-type prefix (MOTION, NOTICE, ORDER …) and
    # its position.  Prefix-matching lines are strongly preferred, and
    # earlier lines (closer to caption) win ties.
    if all(s == 0.0 for _, _, s in scored):
        fallback: list[tuple[IndexedLine, str, float]] = []
        for il, reason, _ in scored:
            has_prefix = bool(_PREFIX_PATTERN.match(il.text.strip()))
            if has_prefix:
                bonus = 1.0
            elif reason == "opening_pattern":
                bonus = 0.8 if has_prefix else 0.5
            else:
                bonus = 0.3
            pos_penalty = il.line_num * 0.005
            fallback.append((il, reason, max(bonus - pos_penalty, 0.0)))
        fallback.sort(key=lambda x: x[2], reverse=True)
        return fallback

    return scored


_RE_SUBJECT = re.compile(
    r"^(?:Re|Subject)\s*[:\.][ ]*(.+)",
    re.IGNORECASE,
)


_RE_BARE = re.compile(r"^(?:Re|Subject)\s*[:\.]\s*$", re.IGNORECASE)

_RE_IN_RE = re.compile(
    r"^In\s+re[:\s]+(.{5,})",
    re.IGNORECASE,
)


def _extract_re_subject(lines: list[IndexedLine]) -> Optional[TitleResult]:
    """Fallback: extract 'Re:' / 'Subject:' / 'In re' subject line as title.

    Handles both single-line ("Re: Subject text here") and multi-line cases
    where "Re:" appears alone and the subject is on subsequent lines.
    Also matches "In re [Case Name]" lines common in legal letters.
    """
    for idx, ln in enumerate(lines):
        text = ln.text.strip()

        # Single-line: "Re: Subject text here"
        m = _RE_SUBJECT.match(text)
        if not m:
            m = _RE_IN_RE.match(text)
        if m:
            subj = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(subj) >= 5:
                return TitleResult(
                    title=subj,
                    confidence=0.55,
                    evidence=[TitleEvidence(
                        source="regex", page=ln.page_num,
                        span_text=text,
                        char_start=ln.char_start,
                        char_end=ln.char_start + len(ln.text),
                        rule_id="re_subject_fallback", score=0.55,
                    )],
                )

        # Multi-line: "Re:" alone, subject on next line(s)
        if _RE_BARE.match(text):
            parts = []
            end_char = ln.char_start + len(ln.text)
            for nxt in lines[idx + 1:]:
                nxt_text = nxt.text.strip()
                if not nxt_text:
                    break
                if re.match(r"^(?:To\s+Whom|Dear\s|PLEASE\s)", nxt_text, re.IGNORECASE):
                    break
                parts.append(nxt_text)
                end_char = nxt.char_start + len(nxt.text)
                if len(" ".join(parts)) > 150:
                    break
            if parts:
                subj = re.sub(r"\s+", " ", " ".join(parts)).strip()
                if len(subj) >= 5:
                    return TitleResult(
                        title=subj,
                        confidence=0.55,
                        evidence=[TitleEvidence(
                            source="regex", page=ln.page_num,
                            span_text=subj,
                            char_start=ln.char_start,
                            char_end=end_char,
                            rule_id="re_subject_multiline", score=0.55,
                        )],
                    )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def demote_party_title(
    title: Optional[str],
    parties: dict[str, list[str]],
) -> bool:
    """
    Return True if `title` is essentially just a party name and should be demoted.

    Uses length-aware scoring: a single party name token appearing inside a
    long legal title (e.g., "NETGEAR" in "NETGEAR INC'S MOTION TO STAY")
    is NOT grounds for demotion.  The title must be *predominantly* composed
    of party-name tokens to qualify.
    """
    if not title:
        return False
    t_upper = re.sub(r"\s+", " ", title).strip().upper()
    if len(t_upper) < 5:
        return False

    _STOP = {"THE", "OF", "AND", "OR", "IN", "FOR", "TO", "A", "AN",
             "INC", "LLC", "LTD", "LP", "CO", "VS", "ET", "AL", "RE"}
    t_tokens = [t for t in re.findall(r"[A-Z]+", t_upper) if t not in _STOP]
    if not t_tokens:
        return False

    for role_names in parties.values():
        for pname in role_names:
            p_upper = re.sub(r"\s+", " ", pname).strip().upper()
            if not p_upper:
                continue

            # Exact or near-exact match (title ≈ party name)
            if t_upper == p_upper:
                return True

            p_tokens = set(re.findall(r"[A-Z]+", p_upper)) - _STOP
            if not p_tokens:
                continue

            overlap_count = len(set(t_tokens) & p_tokens)

            # Require at least 2 overlapping content tokens
            if overlap_count < 2:
                continue

            overlap_ratio = overlap_count / len(t_tokens)

            # For short titles (<=4 content tokens), require >=80% overlap
            # For longer titles, require >=60% — but the title must be
            # *mostly* party name with minimal legal-term padding
            if len(t_tokens) <= 4 and overlap_ratio >= 0.80:
                return True
            if len(t_tokens) > 4 and overlap_ratio >= 0.60:
                return True

    return False


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

    # When the title zone yields no candidates (common with California
    # pleading paper where margin line numbers consume the 60-line budget),
    # extend the search to the full first page.
    if not candidates and zones.first_page_zone:
        extra = [il for il in zones.first_page_zone
                 if il.line_num not in {tl.line_num for tl in title_lines}]
        if extra:
            candidates = _collect_candidates(zones.first_page_zone)

    # Try Re: subject early — if we find one AND there are no strong
    # candidates (typical for letters where the only caps_lines are
    # phone numbers or person names), prefer the Re: subject.
    # Skip this when the sole candidate contains a doc-type keyword
    # (MOTION, MEMORANDUM, etc.) — that's a real title, not noise.
    re_title = _extract_re_subject(title_lines)
    if re_title:
        use_re = False
        if len(candidates) == 0:
            use_re = True
        elif len(candidates) == 1:
            sole_text = candidates[0][0].text.strip()
            word_count = len(sole_text.split())
            has_doctype_kw = any(
                re.search(r"\b" + re.escape(p) + r"\b", sole_text, re.IGNORECASE)
                for p in TITLE_PREFIXES
            )
            # Override with Re: only if the sole candidate looks like a
            # person/entity name (short, no legal keywords) rather than
            # a real document title.
            if not has_doctype_kw and word_count < 4:
                use_re = True
        if use_re:
            return re_title

    if not candidates:
        if re_title:
            return re_title
        return TitleResult(title=None, confidence=0.0, evidence=[])

    ranked = _rank_candidates(candidates, context_text)

    if not ranked:
        if re_title:
            return re_title
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
