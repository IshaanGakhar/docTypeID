"""
Per-word attention scores from Gemma 3 270M for an input string.

Runs a forward pass with output_attentions=True, averages across all
heads and all layers, then maps token-level scores back to words.

Usage:
    python scripts/attention_viz.py
    python scripts/attention_viz.py --text "Filed On: $"
    python scripts/attention_viz.py --text "Filed On: $" --layer 5
    python scripts/attention_viz.py --text "Filed On: $" --layer all --head 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BASE_MODEL = "google/gemma-3-270m-it"

# Try to use the LoRA adapter if it exists
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "gemma3-270m-lora-adapter"


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_scores(
    text: str,
    layer: int | str = "all",   # int → specific layer, "all" → average all layers
    head:  int | str = "all",   # int → specific head,  "all" → average all heads
) -> tuple[list[str], list[float], list[str]]:
    """
    Returns:
        tokens   — list of token strings
        scores   — per-token attention score (averaged as requested)
        words    — list of word strings (tokens merged back)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {BASE_MODEL} …")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    use_adapter = ADAPTER_DIR.exists() and (ADAPTER_DIR / "adapter_config.json").exists()

    print(f"Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",   # eager required for attention output on CPU
    )

    if use_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print(f"  (LoRA adapter loaded from {ADAPTER_DIR})")

    model.eval()

    # ── Tokenize ──────────────────────────────────────────────────────────
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]                        # (1, seq_len)

    token_strs = tok.convert_ids_to_tokens(input_ids[0].tolist())

    # ── Forward pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(
            **enc,
            output_attentions=True,
        )

    # out.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    n_layers = len(out.attentions)
    n_heads  = out.attentions[0].shape[1]

    # ── Select layers ─────────────────────────────────────────────────────
    if layer == "all":
        attn_layers = list(out.attentions)              # all layers
    else:
        idx = int(layer)
        if not (0 <= idx < n_layers):
            raise ValueError(f"Layer {idx} out of range (model has {n_layers} layers)")
        attn_layers = [out.attentions[idx]]

    # Stack → (n_selected_layers, 1, n_heads, seq, seq)
    stacked = torch.stack(attn_layers, dim=0)           # (L, 1, H, S, S)
    stacked = stacked.squeeze(1)                        # (L, H, S, S)

    # ── Select heads ─────────────────────────────────────────────────────
    if head == "all":
        attn = stacked.mean(dim=1)                      # (L, S, S)
    else:
        h = int(head)
        if not (0 <= h < n_heads):
            raise ValueError(f"Head {h} out of range (model has {n_heads} heads)")
        attn = stacked[:, h, :, :]                      # (L, S, S)

    # Average over layers
    attn = attn.mean(dim=0)                             # (S, S)

    # Mean over query dimension → per-key (= per-token) importance
    per_token = attn.mean(dim=0)                        # (S,)
    per_token = per_token / (per_token.sum() + 1e-9)    # normalise to sum=1

    scores = per_token.tolist()

    # ── Map tokens → words ────────────────────────────────────────────────
    # Gemma uses SentencePiece: word-initial tokens start with "▁"
    words:       list[str]   = []
    word_scores: list[float] = []

    current_word  = ""
    current_score = 0.0

    for tok_str, score in zip(token_strs, scores):
        is_new_word = tok_str.startswith("▁") or tok_str in ("<bos>", "<eos>")
        clean = tok_str.lstrip("▁")

        if is_new_word and current_word:
            words.append(current_word)
            word_scores.append(current_score)
            current_word  = clean
            current_score = score
        else:
            current_word  += clean
            current_score += score

    if current_word:
        words.append(current_word)
        word_scores.append(current_score)

    return token_strs, scores, words, word_scores


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _bar(v: float, width: int = 30) -> str:
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


def display(
    text:        str,
    token_strs:  list[str],
    tok_scores:  list[float],
    words:       list[str],
    word_scores: list[float],
    layer, head,
):
    layer_label = f"layer {layer}" if layer != "all" else "all layers"
    head_label  = f"head {head}"   if head  != "all" else "all heads"

    print(f"\n{'═'*60}")
    print(f"  Input : {text!r}")
    print(f"  Model : {BASE_MODEL}")
    print(f"  Scope : {layer_label}, {head_label} (averaged)")
    print(f"{'═'*60}\n")

    # Token-level table
    print("── Token-level attention ───────────────────────────────")
    print(f"  {'#':<4} {'Token':<18} {'Score':>7}  Bar")
    print(f"  {'─'*4} {'─'*18} {'─'*7}  {'─'*30}")
    for i, (t, s) in enumerate(zip(token_strs, tok_scores)):
        print(f"  {i:<4} {repr(t):<18} {s:>7.4f}  {_bar(s)}")

    # Word-level table
    print(f"\n── Word-level attention (tokens merged) ────────────────")
    print(f"  {'Word':<20} {'Score':>7}  Bar")
    print(f"  {'─'*20} {'─'*7}  {'─'*30}")

    max_s = max(word_scores) if word_scores else 1.0
    for w, s in zip(words, word_scores):
        norm = s / max_s         # normalise to [0,1] for bar display
        print(f"  {w:<20} {s:>7.4f}  {_bar(norm)}")

    print()

    # Top-3 words
    ranked = sorted(zip(words, word_scores), key=lambda x: x[1], reverse=True)
    print("── Top attended words ──────────────────────────────────")
    for rank, (w, s) in enumerate(ranked[:3], 1):
        print(f"  #{rank}  {w!r:<20}  {s:.4f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-word attention scores from Gemma 3 270M"
    )
    parser.add_argument("--text",  default="Filed On: $",
                        help="Input string to analyse (default: 'Filed On: $')")
    parser.add_argument("--layer", default="all",
                        help="Layer index or 'all' (default: all)")
    parser.add_argument("--head",  default="all",
                        help="Head index or 'all' (default: all)")
    args = parser.parse_args()

    layer = args.layer if args.layer == "all" else int(args.layer)
    head  = args.head  if args.head  == "all" else int(args.head)

    token_strs, tok_scores, words, word_scores = get_attention_scores(
        text=args.text, layer=layer, head=head
    )
    display(args.text, token_strs, tok_scores, words, word_scores, layer, head)
