"""
Interactive bipartite attention visualizer using Gradio.

Shows how tokens in String A attend to tokens in String B using
Gemma 3 270M (with optional LoRA adapter). Hover over any word to
see its attention connections highlighted with exact scores.

Usage:
    python scripts/attention_ui.py
    # Opens at http://localhost:7860
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import torch

BASE_MODEL  = "google/gemma-3-270m-it"
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "gemma3-270m-lora-adapter"
SEPARATOR   = " [SEP] "

# ---------------------------------------------------------------------------
# Global model cache — loaded once, reused across requests
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_model_and_tok(use_adapter: bool):
    """Return (model, tokenizer), loading from cache or disk."""
    key = "adapter" if use_adapter else "base"
    if key in _CACHE:
        return _CACHE[key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )

    if use_adapter:
        from peft import PeftModel
        if (ADAPTER_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    model.eval()
    _CACHE[key] = (model, tok)
    return model, tok


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------

def _merge_subtokens(token_strs: list[str], matrix, axis: int):
    """
    Merge SentencePiece sub-tokens into words along the given axis of
    the attention matrix by summing weights for merged tokens.

    Returns (words, merged_matrix).
    """
    groups: list[tuple[str, list[int]]] = []
    for i, t in enumerate(token_strs):
        is_new = t.startswith("▁") or t in ("<bos>", "<eos>")
        clean = t.lstrip("▁")
        if is_new or not groups:
            groups.append((clean, [i]))
        else:
            old_word, indices = groups[-1]
            groups[-1] = (old_word + clean, indices + [i])

    words = [w for w, _ in groups]

    merged_rows: list = []
    for _, indices in groups:
        if axis == 0:
            sliced = matrix[indices, :]
        else:
            sliced = matrix[:, indices]
        merged_rows.append(sliced.sum(axis=axis, keepdims=False))

    import numpy as np
    if axis == 0:
        merged = np.stack(merged_rows, axis=0)
    else:
        merged = np.stack(merged_rows, axis=1)

    return words, merged


def compute_cross_attention(
    string_a: str,
    string_b: str,
    use_adapter: bool = False,
    layer: str = "all",
    head: str = "all",
) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    """
    Build token sequence  [BOS] + B + SEP + A  so that A tokens (which come
    last) can attend backwards to B tokens through the causal mask.

    Returns (words_a, words_b, norm_matrix, raw_matrix).
    *norm_matrix* has rows normalised to 1 (for diagram lines / per-word view).
    *raw_matrix* has the raw averaged attention weights (for aggregate scores).
    """
    import numpy as np

    model, tok = _get_model_and_tok(use_adapter)

    # Tokenize each part independently so index boundaries are exact
    ids_b   = tok(string_b, add_special_tokens=False)["input_ids"]
    ids_sep = tok(SEPARATOR, add_special_tokens=False)["input_ids"]
    ids_a   = tok(string_a, add_special_tokens=False)["input_ids"]

    bos     = [tok.bos_token_id]

    # Concatenation order: [BOS] B SEP A
    # A comes last → causal mask lets A attend to B
    full_ids = bos + ids_b + ids_sep + ids_a

    b_start = len(bos)
    b_end   = b_start + len(ids_b)
    a_start = b_end + len(ids_sep)
    a_end   = a_start + len(ids_a)

    input_ids  = torch.tensor([full_ids])
    attn_mask  = torch.ones_like(input_ids)
    all_tokens = tok.convert_ids_to_tokens(full_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_attentions=True)

    n_layers = len(out.attentions)
    n_heads  = out.attentions[0].shape[1]

    if layer == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(layer)]

    if head == "all":
        heads = list(range(n_heads))
    else:
        heads = [int(head)]

    stacked = torch.stack([out.attentions[l] for l in layers], dim=0)
    stacked = stacked.squeeze(1)          # (L, H, S, S)
    stacked = stacked[:, heads, :, :]     # (L, selected_H, S, S)
    attn = stacked.mean(dim=(0, 1))       # (S, S) averaged

    a_indices = list(range(a_start, a_end))
    b_indices = list(range(b_start, b_end))

    if not a_indices or not b_indices:
        return ["(empty)"], ["(empty)"], [[0.0]], [[0.0]]

    a_tokens = [all_tokens[i] for i in a_indices]
    b_tokens = [all_tokens[i] for i in b_indices]

    # Cross-attention: rows = A queries, cols = B keys
    cross_raw = attn[a_indices][:, b_indices].numpy()  # (len_A_tok, len_B_tok)

    # Merge sub-tokens into words (on the raw scores)
    words_a, cross_raw = _merge_subtokens(a_tokens, cross_raw, axis=0)
    words_b, cross_raw = _merge_subtokens(b_tokens, cross_raw, axis=1)

    # Normalised copy (rows sum to 1) for diagram lines / per-word view
    cross_norm = cross_raw.copy()
    row_sums = cross_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cross_norm = cross_norm / row_sums

    return words_a, words_b, cross_norm.tolist(), cross_raw.tolist()


# ---------------------------------------------------------------------------
# SVG rendering with inline JS for hover
# ---------------------------------------------------------------------------

def render_diagram(
    words_a: list[str],
    words_b: list[str],
    matrix: list[list[float]],
    raw_matrix: list[list[float]] | None = None,
    sel_side: str | None = None,
    sel_idx: int | None = None,
) -> str:
    """Render the full HTML: SVG diagram + optional score table.

    *matrix* is row-normalised (for lines / per-word detail).
    *raw_matrix* has raw attention weights (for aggregate summary tables).
    """
    if raw_matrix is None:
        raw_matrix = matrix

    n_a = len(words_a)
    n_b = len(words_b)

    import numpy as np
    mat = np.array(matrix)
    global_max = float(mat.max()) if mat.size else 1.0
    if global_max == 0:
        global_max = 1.0

    row_h     = 36
    pad_top   = 60
    pad_bot   = 30
    score_w   = 65
    col_left  = 180 + score_w
    col_right = 520 + score_w
    svg_w     = 720 + score_w * 2
    svg_h     = pad_top + max(n_a, n_b) * row_h + pad_bot

    has_sel = sel_side is not None and sel_idx is not None

    def y_pos(idx, total):
        block_h = total * row_h
        offset = (svg_h - pad_top - pad_bot - block_h) / 2
        return pad_top + offset + idx * row_h + row_h / 2

    # ── lines ──
    lines_svg = []
    for i in range(n_a):
        for j in range(n_b):
            score = matrix[i][j]
            ya = y_pos(i, n_a)
            yb = y_pos(j, n_b)
            if has_sel:
                is_active = ((sel_side == "a" and i == sel_idx)
                             or (sel_side == "b" and j == sel_idx))
                if is_active:
                    raw = score / global_max
                    opacity = max(raw ** 0.5 * 0.95, 0.18)
                    color, sw = "#ff9944", "3.5"
                else:
                    opacity, color, sw = 0.03, "#4a90d9", "1"
            else:
                raw = score / global_max
                opacity = max(raw ** 0.5 * 0.9, 0.04)
                color, sw = "#4a90d9", "2"
            lines_svg.append(
                f'<line x1="{col_left + 10}" y1="{ya:.1f}" '
                f'x2="{col_right - 10}" y2="{yb:.1f}" '
                f'stroke="{color}" stroke-width="{sw}" opacity="{opacity:.4f}" />'
            )

    # ── word labels (A = left) ──
    words_a_svg = []
    for i, w in enumerate(words_a):
        ya = y_pos(i, n_a)
        esc = html.escape(w)
        is_sel = has_sel and sel_side == "a" and i == sel_idx
        fill = "#ffffff" if is_sel else "#e0e0e0"
        fw   = "bold" if is_sel else "normal"
        row_score = sum(matrix[i])
        bg_opacity = min(row_score / (global_max * n_b) * 2, 0.6)
        words_a_svg.append(
            f'<rect x="{col_left - 150}" y="{ya - 14}" width="155" height="28" '
            f'rx="4" fill="rgba(74,144,217,{bg_opacity:.2f})" />'
            f'<text x="{col_left - 5}" y="{ya + 5}" text-anchor="end" '
            f'font-size="14" font-family="monospace" fill="{fill}" '
            f'font-weight="{fw}">{esc}</text>'
        )

    # ── word labels (B = right) ──
    words_b_svg = []
    for j, w in enumerate(words_b):
        yb = y_pos(j, n_b)
        esc = html.escape(w)
        is_sel = has_sel and sel_side == "b" and j == sel_idx
        fill = "#ffffff" if is_sel else "#e0e0e0"
        fw   = "bold" if is_sel else "normal"
        col_score = sum(matrix[i][j] for i in range(n_a))
        bg_opacity = min(col_score / (global_max * n_a) * 2, 0.6)
        words_b_svg.append(
            f'<rect x="{col_right - 5}" y="{yb - 14}" width="155" height="28" '
            f'rx="4" fill="rgba(217,144,74,{bg_opacity:.2f})" />'
            f'<text x="{col_right + 5}" y="{yb + 5}" text-anchor="start" '
            f'font-size="14" font-family="monospace" fill="{fill}" '
            f'font-weight="{fw}">{esc}</text>'
        )

    # ── inline score annotations when a word is selected ──
    score_labels_svg = []
    if has_sel:
        if sel_side == "a":
            for j in range(n_b):
                pct = f"{matrix[sel_idx][j] * 100:.1f}%"
                yb = y_pos(j, n_b) + 4
                score_labels_svg.append(
                    f'<text x="{col_right + 160}" y="{yb}" text-anchor="start" '
                    f'font-size="11" font-family="monospace" fill="#ff9944">{pct}</text>'
                )
        else:
            for i in range(n_a):
                pct = f"{matrix[i][sel_idx] * 100:.1f}%"
                ya = y_pos(i, n_a) + 4
                score_labels_svg.append(
                    f'<text x="{col_left - 160}" y="{ya}" text-anchor="end" '
                    f'font-size="11" font-family="monospace" fill="#ff9944">{pct}</text>'
                )

    title_a = "String A" + ("  (selected)" if has_sel and sel_side == "a" else "")
    title_b = "String B" + ("  (selected)" if has_sel and sel_side == "b" else "")

    out = f"""\
<div style="font-family:monospace;">
<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg"
     style="background:#1e1e2e; border-radius:8px; display:block;">
  <text x="{col_left - 70}" y="30" text-anchor="middle"
        font-size="13" font-weight="bold" fill="#8888cc"
        font-family="sans-serif">{title_a}</text>
  <text x="{col_right + 70}" y="30" text-anchor="middle"
        font-size="13" font-weight="bold" fill="#cc8844"
        font-family="sans-serif">{title_b}</text>
  {''.join(lines_svg)}
  {''.join(words_a_svg)}
  {''.join(words_b_svg)}
  {''.join(score_labels_svg)}
</svg>
"""

    # ── detail table ──
    if has_sel:
        if sel_side == "a":
            heading = (f'<span style="color:#8888cc;font-weight:bold">'
                       f'{html.escape(words_a[sel_idx])}</span> '
                       f'&#8594; attends to:')
            pairs = [(words_b[j], matrix[sel_idx][j]) for j in range(n_b)]
            pair_color = "#cc8844"
        else:
            heading = (f'<span style="color:#cc8844;font-weight:bold">'
                       f'{html.escape(words_b[sel_idx])}</span> '
                       f'&#8592; attended by:')
            pairs = [(words_a[i], matrix[i][sel_idx]) for i in range(n_a)]
            pair_color = "#8888cc"

        pairs.sort(key=lambda x: -x[1])
        max_s = pairs[0][1] if pairs else 1.0

        rows_html = []
        for word, score in pairs:
            pct = f"{score * 100:.1f}"
            bar_w = int(score / max_s * 200) if max_s else 0
            rows_html.append(
                f'<tr style="border-bottom:1px solid #222">'
                f'<td style="padding:5px 10px;color:{pair_color}">{html.escape(word)}</td>'
                f'<td style="padding:5px 10px">'
                f'<div style="background:#333;border-radius:3px;height:16px;width:200px">'
                f'<div style="background:#ff9944;border-radius:3px;height:16px;'
                f'width:{bar_w}px"></div></div></td>'
                f'<td style="padding:5px 10px;text-align:right;color:#eee;'
                f'font-weight:bold">{pct}%</td></tr>'
            )

        out += f"""\
<div style="margin-top:10px; background:#181825; border:1px solid #44446688;
            border-radius:8px; padding:14px 18px; color:#ddd; font-size:13px;
            max-width:{svg_w}px; box-sizing:border-box;">
  <div style="margin-bottom:10px;font-size:15px">{heading}</div>
  <table style="border-collapse:collapse; width:100%">
    <tr style="color:#888; border-bottom:1px solid #333">
      <th style="text-align:left;padding:5px 10px">Word</th>
      <th style="text-align:left;padding:5px 10px">Attention</th>
      <th style="text-align:right;padding:5px 10px">Score</th></tr>
    {''.join(rows_html)}
  </table>
</div>
"""

    # ── always-visible summary: total attention per word ──
    import numpy as _np

    _raw = _np.array(raw_matrix)
    row_sums_raw = _raw.sum(axis=1)   # total raw attention each A word gives to B
    row_max_raw  = float(row_sums_raw.max()) if row_sums_raw.size else 1.0
    col_sums_raw = _raw.sum(axis=0)   # total raw attention each B word receives
    col_max_raw  = float(col_sums_raw.max()) if col_sums_raw.size else 1.0

    summary_rows = []
    sorted_a = sorted(range(n_a), key=lambda i: -row_sums_raw[i])
    for i in sorted_a:
        total = float(row_sums_raw[i])
        bar_w = int(total / row_max_raw * 200) if row_max_raw else 0
        esc_a = html.escape(words_a[i])
        summary_rows.append(
            f'<tr style="border-bottom:1px solid #222">'
            f'<td style="padding:5px 10px;color:#8888cc">{esc_a}</td>'
            f'<td style="padding:5px 10px">'
            f'<div style="background:#333;border-radius:3px;height:16px;width:200px">'
            f'<div style="background:#4a90d9;border-radius:3px;height:16px;'
            f'width:{bar_w}px"></div></div></td>'
            f'<td style="padding:5px 10px;text-align:right;color:#eee;'
            f'font-weight:bold">{total:.4f}</td></tr>'
        )

    recv_rows = []
    sorted_b = sorted(range(n_b), key=lambda j: -col_sums_raw[j])
    for j in sorted_b:
        esc_b = html.escape(words_b[j])
        total = float(col_sums_raw[j])
        bar_w = int(total / col_max_raw * 200) if col_max_raw else 0
        recv_rows.append(
            f'<tr style="border-bottom:1px solid #222">'
            f'<td style="padding:5px 10px;color:#cc8844">{esc_b}</td>'
            f'<td style="padding:5px 10px">'
            f'<div style="background:#333;border-radius:3px;height:16px;width:200px">'
            f'<div style="background:#4a90d9;border-radius:3px;height:16px;'
            f'width:{bar_w}px"></div></div></td>'
            f'<td style="padding:5px 10px;text-align:right;color:#eee;'
            f'font-weight:bold">{total:.4f}</td></tr>'
        )

    out += f"""\
<div style="margin-top:14px; display:flex; gap:14px; max-width:{svg_w}px; flex-wrap:wrap;">
  <div style="flex:1; min-width:280px; background:#181825; border:1px solid #44446688;
              border-radius:8px; padding:14px 18px; color:#ddd; font-size:13px;">
    <div style="margin-bottom:10px;font-size:14px;font-weight:bold;color:#8888cc">
      Raw attention given by each word in String A (sum over all B words)</div>
    <table style="border-collapse:collapse; width:100%">
      <tr style="color:#888; border-bottom:1px solid #333">
        <th style="text-align:left;padding:5px 10px">Word (A)</th>
        <th style="text-align:left;padding:5px 10px">Attention</th>
        <th style="text-align:right;padding:5px 10px">Raw Score</th></tr>
      {''.join(summary_rows)}
    </table>
  </div>
  <div style="flex:1; min-width:280px; background:#181825; border:1px solid #44446688;
              border-radius:8px; padding:14px 18px; color:#ddd; font-size:13px;">
    <div style="margin-bottom:10px;font-size:14px;font-weight:bold;color:#cc8844">
      Raw attention received by each word in String B (sum over all A words)</div>
    <table style="border-collapse:collapse; width:100%">
      <tr style="color:#888; border-bottom:1px solid #333">
        <th style="text-align:left;padding:5px 10px">Word (B)</th>
        <th style="text-align:left;padding:5px 10px">Attention</th>
        <th style="text-align:right;padding:5px 10px">Raw Score</th></tr>
      {''.join(recv_rows)}
    </table>
  </div>
</div>
"""

    out += "</div>"
    return out


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr

    adapter_available = (ADAPTER_DIR / "adapter_config.json").exists()
    model_choices = ["Base Gemma 270M"]
    if adapter_available:
        model_choices.append("Gemma 270M + LoRA")

    layer_choices = ["all"] + [str(i) for i in range(18)]
    head_choices  = ["all"] + [str(i) for i in range(4)]

    # ── callbacks ────────────────────────────────────────────────

    def on_compute(string_a, string_b, model_choice, layer, head):
        """Run attention, render default diagram, populate word selectors."""
        if not string_a.strip() or not string_b.strip():
            return ("<p style='color:#ff6666'>Both strings required.</p>",
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    [], [], [], [])

        use_adapter = model_choice == "Gemma 270M + LoRA"
        try:
            wa, wb, mat_norm, mat_raw = compute_cross_attention(
                string_a, string_b,
                use_adapter=use_adapter, layer=layer, head=head,
            )
        except Exception as e:
            return (f"<p style='color:#ff6666'>Error: {html.escape(str(e))}</p>",
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    [], [], [], [])

        choices_a = [f"A: {w}" for w in wa]
        choices_b = [f"B: {w}" for w in wb]
        diagram   = render_diagram(wa, wb, mat_norm, mat_raw)

        return (diagram,
                gr.update(choices=choices_a, value=None, visible=True),
                gr.update(choices=choices_b, value=None, visible=True),
                wa, wb, mat_norm, mat_raw)

    def on_select_a(choice, words_a, words_b, matrix, raw_matrix):
        """User picked a word from String A."""
        if not matrix or choice is None:
            return render_diagram(words_a, words_b, matrix, raw_matrix)
        idx = next(
            (i for i, w in enumerate(words_a) if f"A: {w}" == choice), None
        )
        if idx is None:
            return render_diagram(words_a, words_b, matrix, raw_matrix)
        return render_diagram(words_a, words_b, matrix, raw_matrix,
                              sel_side="a", sel_idx=idx)

    def on_select_b(choice, words_a, words_b, matrix, raw_matrix):
        """User picked a word from String B."""
        if not matrix or choice is None:
            return render_diagram(words_a, words_b, matrix, raw_matrix)
        idx = next(
            (j for j, w in enumerate(words_b) if f"B: {w}" == choice), None
        )
        if idx is None:
            return render_diagram(words_a, words_b, matrix, raw_matrix)
        return render_diagram(words_a, words_b, matrix, raw_matrix,
                              sel_side="b", sel_idx=idx)

    # ── layout ───────────────────────────────────────────────────

    with gr.Blocks(
        title="Attention Visualizer",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as app:
        gr.Markdown("## Bipartite Attention Visualizer — Gemma 3 270M")
        gr.Markdown(
            "Enter two strings, hit **Compute**, then **click a word** "
            "from either dropdown to see its attention scores to every word "
            "in the other string."
        )

        with gr.Row():
            string_a = gr.Textbox(
                label="String A",
                value="FILED: NEW YORK COUNTY CLERK 03/22/2021 03:01 PM",
                lines=2,
            )
            string_b = gr.Textbox(
                label="String B",
                value="when was it filed?",
                lines=2,
            )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=model_choices, value=model_choices[0], label="Model",
            )
            layer_dd = gr.Dropdown(
                choices=layer_choices, value="all",
                label="Layer (0\u201317 or all)",
            )
            head_dd = gr.Dropdown(
                choices=head_choices, value="all",
                label="Head (0\u20133 or all)",
            )

        run_btn = gr.Button("Compute Attention", variant="primary")

        output = gr.HTML(label="Attention Diagram")

        # Hidden state to hold computed results across callbacks
        st_wa      = gr.State([])
        st_wb      = gr.State([])
        st_mat     = gr.State([])
        st_mat_raw = gr.State([])

        gr.Markdown("### Inspect a word")
        with gr.Row():
            sel_a = gr.Dropdown(
                choices=[], value=None, label="Select word from String A",
                visible=False, interactive=True,
            )
            sel_b = gr.Dropdown(
                choices=[], value=None, label="Select word from String B",
                visible=False, interactive=True,
            )

        # Wire events
        run_btn.click(
            fn=on_compute,
            inputs=[string_a, string_b, model_dd, layer_dd, head_dd],
            outputs=[output, sel_a, sel_b, st_wa, st_wb, st_mat, st_mat_raw],
        )
        sel_a.change(
            fn=on_select_a,
            inputs=[sel_a, st_wa, st_wb, st_mat, st_mat_raw],
            outputs=output,
        )
        sel_b.change(
            fn=on_select_b,
            inputs=[sel_b, st_wa, st_wb, st_mat, st_mat_raw],
            outputs=output,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
