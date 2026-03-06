"""
CRF-based Named Entity Recognition for legal documents.
Entities: COURT, JUDGE, DATE, CASE_NO, PARTY.

Training: scripts/train_crf.py
Inference: extract_entities_crf()

Falls back to empty output gracefully if model file is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.config import CRF_CONTEXT_WINDOW, CRF_ENTITY_LABELS, MODEL_CRF_PATH


# ---------------------------------------------------------------------------
# Token shape helper
# ---------------------------------------------------------------------------

def _token_shape(token: str) -> str:
    """
    Encode a token's character-class pattern, e.g. "Smith" → "Aa+" or "123" → "d+".
    """
    if not token:
        return ""
    shape = []
    prev = None
    for ch in token:
        if ch.isupper():
            c = "A"
        elif ch.islower():
            c = "a"
        elif ch.isdigit():
            c = "d"
        else:
            c = ch
        if c != prev:
            shape.append(c)
            prev = c
        else:
            if shape and not shape[-1].endswith("+"):
                shape[-1] = shape[-1] + "+"
    return "".join(shape)


def _digit_pattern(token: str) -> str:
    if re.fullmatch(r"\d{4}", token):
        return "year"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", token):
        return "date_slash"
    if re.fullmatch(r"\d{1,2}-\d{2,4}", token):
        return "partial_date"
    if re.search(r"\d", token):
        return "has_digit"
    return "no_digit"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _word_features(tokens: list[str], i: int) -> dict[str, object]:
    """
    Build feature dict for token at position i within its sentence.
    """
    token = tokens[i]
    t_lower = token.lower()

    feats: dict[str, object] = {
        "bias": 1.0,
        "token.lower": t_lower,
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.prefix3": t_lower[:3],
        "token.suffix3": t_lower[-3:],
        "token.prefix2": t_lower[:2],
        "token.suffix2": t_lower[-2:],
        "token.shape": _token_shape(token),
        "token.digit_pattern": _digit_pattern(token),
        "token.len_bucket": min(len(token) // 3, 5),
        "token.pos_bucket": min(i // 5, 10),
        "token.is_punct": not token.isalnum(),
    }

    # Context window
    for delta in range(1, CRF_CONTEXT_WINDOW + 1):
        # Previous tokens
        prev_idx = i - delta
        if prev_idx >= 0:
            prev = tokens[prev_idx]
            feats[f"-{delta}:token.lower"] = prev.lower()
            feats[f"-{delta}:token.istitle"] = prev.istitle()
            feats[f"-{delta}:token.isupper"] = prev.isupper()
            feats[f"-{delta}:token.shape"] = _token_shape(prev)
        else:
            feats[f"BOS_{delta}"] = True

        # Next tokens
        next_idx = i + delta
        if next_idx < len(tokens):
            nxt = tokens[next_idx]
            feats[f"+{delta}:token.lower"] = nxt.lower()
            feats[f"+{delta}:token.istitle"] = nxt.istitle()
            feats[f"+{delta}:token.isupper"] = nxt.isupper()
            feats[f"+{delta}:token.shape"] = _token_shape(nxt)
        else:
            feats[f"EOS_{delta}"] = True

    return feats


def sent_to_features(tokens: list[str]) -> list[dict]:
    return [_word_features(tokens, i) for i in range(len(tokens))]


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class CRFSpan:
    label: str
    tokens: list[str]
    start_token: int
    end_token: int   # exclusive


@dataclass
class CRFResult:
    spans: list[CRFSpan] = field(default_factory=list)
    raw_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _collect_spans(tokens: list[str], labels: list[str]) -> list[CRFSpan]:
    """Convert BIO-tagged token list to entity spans."""
    spans: list[CRFSpan] = []
    current_label: Optional[str] = None
    current_tokens: list[str] = []
    current_start = 0

    for i, (tok, lbl) in enumerate(zip(tokens, labels)):
        bio_prefix = lbl[:2] if len(lbl) > 2 else ""
        entity = lbl[2:] if bio_prefix in ("B-", "I-") else lbl

        if bio_prefix == "B-":
            if current_label is not None:
                spans.append(CRFSpan(
                    label=current_label,
                    tokens=current_tokens,
                    start_token=current_start,
                    end_token=i,
                ))
            current_label = entity
            current_tokens = [tok]
            current_start = i
        elif bio_prefix == "I-" and current_label == entity:
            current_tokens.append(tok)
        else:
            if current_label is not None:
                spans.append(CRFSpan(
                    label=current_label,
                    tokens=current_tokens,
                    start_token=current_start,
                    end_token=i,
                ))
            current_label = None
            current_tokens = []

    if current_label is not None:
        spans.append(CRFSpan(
            label=current_label,
            tokens=current_tokens,
            start_token=current_start,
            end_token=len(tokens),
        ))

    return spans


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[\w\.\-\/:]+|[^\w\s]", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities_crf(
    text: str,
    page_num: int = 1,
) -> CRFResult:
    """
    Run CRF NER on text. Returns CRFResult with entity spans.
    If model file is missing, returns empty CRFResult.
    """
    if not MODEL_CRF_PATH.exists():
        return CRFResult()

    try:
        import joblib
        crf = joblib.load(MODEL_CRF_PATH)
    except Exception:
        return CRFResult()

    tokens = tokenize(text)
    if not tokens:
        return CRFResult()

    features = sent_to_features(tokens)

    try:
        labels = crf.predict([features])[0]
    except Exception:
        return CRFResult()

    spans = _collect_spans(tokens, labels)
    return CRFResult(spans=spans, raw_labels=labels)


def get_entities_by_label(result: CRFResult, label: str) -> list[str]:
    """Return text for all spans with the given entity label."""
    return [
        " ".join(span.tokens)
        for span in result.spans
        if span.label == label
    ]
