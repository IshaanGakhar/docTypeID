"""
Text preprocessing: whitespace normalization, zone splitting, and
line-with-page-number indexing.

Header/footer stripping is intentionally OFF by default (STRIP_REPEATED_LINES_IN_BODY=False).
Rationale: legal documents repeat the case caption (court name, case number,
party names) in ECF running headers on every page — these are exactly the
fields we want to extract. Stripping them would silently drop evidence.

When stripping IS enabled (via config), it only applies to body_zone
(lines after the first TITLE_ZONE_LINES), never to caption or title zones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.config import (
    CAPTION_ZONE_LINES,
    TITLE_ZONE_LINES,
    STRIP_REPEATED_LINES_IN_BODY,
    REPEATED_LINE_MIN_PAGES,
)
from pipeline.pdf_loader import LoadedDocument


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IndexedLine:
    """A single line of text annotated with its source page number."""
    line_num: int    # 0-indexed within the document
    page_num: int    # 1-indexed PDF page
    text: str
    char_start: int  # character offset in full_text
    is_repeated: bool = False  # True when line appeared in header/footer band


@dataclass
class DocumentZones:
    all_lines: list[IndexedLine]
    caption_zone: list[IndexedLine]   # first ~40 lines — never stripped
    title_zone: list[IndexedLine]     # first ~60 lines — never stripped
    body_zone: list[IndexedLine]      # everything after title_zone
    first_page_zone: list[IndexedLine]
    full_text_clean: str              # whitespace-normalised full text
    first_page_blocks: list[dict] = field(default_factory=list)  # PDF block-level bboxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MULTI_SPACE  = re.compile(r"[ \t]{2,}")
_MULTI_NL     = re.compile(r"\n{3,}")
_FORM_FEED    = re.compile(r"\f")
_BULLET_CHARS = re.compile(r"^[\u2022\u2023\u25e6\u2043\u2219]\s*")

_OCR_ARTIFACTS = [
    (re.compile(r"\bl\b(?=\s+\w)"), "I"),   # isolated "l" → "I"
    (re.compile(r"\bO\b(?=\s+\d)"), "0"),   # isolated "O" before digit → "0"
]


def _normalize_whitespace(text: str) -> str:
    text = _FORM_FEED.sub("\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _fix_artifacts(text: str) -> str:
    for pattern, replacement in _OCR_ARTIFACTS:
        text = pattern.sub(replacement, text)
    return text


def _clean_line(line: str) -> str:
    line = _BULLET_CHARS.sub("", line)
    return line.strip()


def _detect_repeated_lines(pages: list, min_pages: int) -> set[str]:
    """
    Return lines that appear verbatim on >= min_pages distinct pages.
    Only called when STRIP_REPEATED_LINES_IN_BODY is True.
    """
    from collections import Counter
    counts: Counter = Counter()
    for page in pages:
        seen_this_page: set[str] = set()
        for raw in page.text.splitlines():
            s = raw.strip()
            if 4 <= len(s) <= 120 and s not in seen_this_page:
                counts[s] += 1
                seen_this_page.add(s)
    return {line for line, n in counts.items() if n >= min_pages}


def _build_indexed_lines(
    pages: list,
    repeated_lines: set[str],
) -> list[IndexedLine]:
    """
    Flatten all pages into IndexedLine objects.
    Repeated lines are tagged with is_repeated=True but NOT dropped —
    callers decide whether to use or skip them.
    """
    indexed: list[IndexedLine] = []
    global_line_num = 0
    global_char_pos = 0

    for page in pages:
        for raw_line in page.text.splitlines():
            cleaned = _clean_line(raw_line)
            if not cleaned:
                global_char_pos += 1
                continue
            is_rep = cleaned in repeated_lines
            indexed.append(IndexedLine(
                line_num=global_line_num,
                page_num=page.page_num,
                text=cleaned,
                char_start=global_char_pos,
                is_repeated=is_rep,
            ))
            global_line_num += 1
            global_char_pos += len(raw_line) + 1

    return indexed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(doc: LoadedDocument) -> DocumentZones:
    """
    Preprocess a LoadedDocument into structured DocumentZones.

    Caption and title zones (first 60 lines) are NEVER filtered — they contain
    the court name, case number, and party names that may repeat across pages.

    Body zone repeated-line stripping is opt-in via config
    (STRIP_REPEATED_LINES_IN_BODY, default False).
    """
    pages = doc.pages

    # Identify repeated lines (tagged but not dropped at this stage)
    repeated_lines: set[str] = set()
    if STRIP_REPEATED_LINES_IN_BODY and len(pages) >= REPEATED_LINE_MIN_PAGES:
        repeated_lines = _detect_repeated_lines(pages, REPEATED_LINE_MIN_PAGES)

    all_lines = _build_indexed_lines(pages, repeated_lines)

    # Caption and title zones: always use all lines, ignoring is_repeated
    caption_zone    = all_lines[:CAPTION_ZONE_LINES]
    title_zone      = all_lines[:TITLE_ZONE_LINES]
    first_page_zone = [il for il in all_lines if il.page_num == 1]

    # Body zone: optionally skip repeated lines
    raw_body = all_lines[TITLE_ZONE_LINES:]
    body_zone = (
        [il for il in raw_body if not il.is_repeated]
        if STRIP_REPEATED_LINES_IN_BODY
        else raw_body
    )

    full_text_clean = _normalize_whitespace(
        _fix_artifacts("\n".join(il.text for il in all_lines))
    )

    return DocumentZones(
        all_lines=all_lines,
        caption_zone=caption_zone,
        title_zone=title_zone,
        body_zone=body_zone,
        first_page_zone=first_page_zone,
        full_text_clean=full_text_clean,
        first_page_blocks=doc.page_blocks if doc.page_blocks else [],
    )


def lines_to_text(lines: list[IndexedLine]) -> str:
    return "\n".join(il.text for il in lines)
