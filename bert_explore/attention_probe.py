"""
Core BERT attention engine.

Loads a BERT model, computes bidirectional self-attention, and extracts
the query-to-document cross-attention submatrix.

Usage::

    from bert_explore.attention_probe import compute_cross_attention

    words_q, words_d, norm, raw = compute_cross_attention(
        query="when was it filed?",
        document="FILED: NEW YORK COUNTY CLERK 03/22/2021 03:01 PM",
    )
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch

DEFAULT_MODEL = "bert-base-uncased"
LEGAL_MODEL = "nlpaueb/legal-bert-base-uncased"

_CACHE: dict = {}


def get_model_and_tok(model_name: str = DEFAULT_MODEL):
    """Return (model, tokenizer), loading from cache or disk."""
    if model_name in _CACHE:
        return _CACHE[model_name]

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
    )
    model.eval()
    _CACHE[model_name] = (model, tok)
    return model, tok


def model_info(model_name: str = DEFAULT_MODEL) -> dict:
    """Return n_layers and n_heads for the loaded model."""
    model, _ = get_model_and_tok(model_name)
    cfg = model.config
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "hidden_size": cfg.hidden_size,
        "model_name": model_name,
    }


def _merge_wordpiece(token_strs: list[str], matrix: np.ndarray, axis: int):
    """Merge WordPiece sub-tokens (##suffix) into whole words along *axis*.

    Sub-token attention weights are summed when merging.
    Returns (words, merged_matrix).
    """
    groups: list[tuple[str, list[int]]] = []
    for i, t in enumerate(token_strs):
        if t.startswith("##"):
            if groups:
                old_word, indices = groups[-1]
                groups[-1] = (old_word + t[2:], indices + [i])
            else:
                groups.append((t[2:], [i]))
        else:
            groups.append((t, [i]))

    words = [w for w, _ in groups]

    merged_rows: list[np.ndarray] = []
    for _, indices in groups:
        if axis == 0:
            sliced = matrix[indices, :]
        else:
            sliced = matrix[:, indices]
        merged_rows.append(sliced.sum(axis=axis, keepdims=False))

    if axis == 0:
        merged = np.stack(merged_rows, axis=0)
    else:
        merged = np.stack(merged_rows, axis=1)

    return words, merged


def compute_cross_attention(
    query: str,
    document: str,
    model_name: str = DEFAULT_MODEL,
    layer: str = "all",
    head: str = "all",
) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    """Compute bidirectional cross-attention between *query* and *document*.

    Tokenises as ``[CLS] query [SEP] document [SEP]`` and extracts the
    query-to-document attention submatrix.  Because BERT is bidirectional,
    every query token can attend to every document token without causal mask
    tricks.

    Returns ``(words_query, words_doc, norm_matrix, raw_matrix)``.

    * *norm_matrix*: rows normalised to sum to 1 (per-word view).
    * *raw_matrix*: raw averaged attention weights (aggregate scores).
    """
    model, tok = get_model_and_tok(model_name)

    enc_q = tok(query, add_special_tokens=False)["input_ids"]
    enc_d = tok(document, add_special_tokens=False)["input_ids"]

    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id

    # [CLS] query [SEP] document [SEP]
    full_ids = [cls_id] + enc_q + [sep_id] + enc_d + [sep_id]

    q_start = 1
    q_end = 1 + len(enc_q)
    d_start = q_end + 1
    d_end = d_start + len(enc_d)

    input_ids = torch.tensor([full_ids])
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[0, d_start:] = 1
    attn_mask = torch.ones_like(input_ids)

    all_tokens = tok.convert_ids_to_tokens(full_ids)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask,
        )

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]

    layers = list(range(n_layers)) if layer == "all" else [int(layer)]
    heads = list(range(n_heads)) if head == "all" else [int(head)]

    stacked = torch.stack([out.attentions[l] for l in layers], dim=0)
    stacked = stacked.squeeze(1)           # (L, H, S, S)
    stacked = stacked[:, heads, :, :]      # (L, sel_H, S, S)
    attn = stacked.mean(dim=(0, 1))        # (S, S)

    q_indices = list(range(q_start, q_end))
    d_indices = list(range(d_start, d_end))

    if not q_indices or not d_indices:
        return ["(empty)"], ["(empty)"], [[0.0]], [[0.0]]

    q_tokens = [all_tokens[i] for i in q_indices]
    d_tokens = [all_tokens[i] for i in d_indices]

    cross_raw = attn[q_indices][:, d_indices].numpy()

    words_q, cross_raw = _merge_wordpiece(q_tokens, cross_raw, axis=0)
    words_d, cross_raw = _merge_wordpiece(d_tokens, cross_raw, axis=1)

    cross_norm = cross_raw.copy()
    row_sums = cross_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cross_norm = cross_norm / row_sums

    return words_q, words_d, cross_norm.tolist(), cross_raw.tolist()


def compute_full_attention(
    query: str,
    document: str,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Return the full per-head attention tensor plus token metadata.

    Returns a dict with:
        ``attn``         — ndarray of shape (n_layers, n_heads, S, S)
        ``all_tokens``   — list of token strings
        ``q_start/q_end``— index range for query tokens
        ``d_start/d_end``— index range for document tokens
    """
    model, tok = get_model_and_tok(model_name)

    enc_q = tok(query, add_special_tokens=False)["input_ids"]
    enc_d = tok(document, add_special_tokens=False)["input_ids"]

    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id

    full_ids = [cls_id] + enc_q + [sep_id] + enc_d + [sep_id]

    q_start = 1
    q_end = 1 + len(enc_q)
    d_start = q_end + 1
    d_end = d_start + len(enc_d)

    input_ids = torch.tensor([full_ids])
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[0, d_start:] = 1
    attn_mask = torch.ones_like(input_ids)

    all_tokens = tok.convert_ids_to_tokens(full_ids)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask,
        )

    attn_tensor = torch.stack(out.attentions, dim=0).squeeze(1).numpy()

    return {
        "attn": attn_tensor,
        "all_tokens": all_tokens,
        "q_start": q_start,
        "q_end": q_end,
        "d_start": d_start,
        "d_end": d_end,
    }
