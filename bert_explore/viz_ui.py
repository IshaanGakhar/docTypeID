"""
Interactive bidirectional attention visualiser for BERT using Gradio.

Shows how query tokens attend to document tokens (and vice versa) using
BERT's full bidirectional self-attention.  Includes entity-type overlays
to visually confirm head specialisation.

Usage::

    python -m bert_explore.viz_ui
    # Opens at http://localhost:7861
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np

from bert_explore.attention_probe import (
    DEFAULT_MODEL,
    LEGAL_MODEL,
    compute_cross_attention,
    model_info,
)

_ENTITY_COLORS = {
    "name":   ("#4a90d9", "name"),
    "date":   ("#50b848", "date"),
    "number": ("#e8a838", "number"),
}

_DATE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"\d{2,}[-:]\w+[-:]\d+"
    r"|\d{2}[a-z]{2,3}\d{3,}"
    r"|\b\d{3,}\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def _token_entity_type(tok: str) -> str | None:
    """Classify a word token by entity type for colour overlay."""
    if _DATE_RE.search(tok):
        return "date"
    if _NUMBER_RE.search(tok):
        return "number"
    if _NAME_RE.match(tok):
        return "name"
    if tok.isupper() and len(tok) > 2 and tok.isalpha():
        return "name"
    return None


def render_diagram(
    words_a: list[str],
    words_b: list[str],
    matrix_ab: list[list[float]],
    raw_ab: list[list[float]] | None = None,
    matrix_ba: list[list[float]] | None = None,
    raw_ba: list[list[float]] | None = None,
    sel_side: str | None = None,
    sel_idx: int | None = None,
    direction: str = "a_to_b",
    show_entity_colors: bool = True,
) -> str:
    """Render the full HTML: SVG diagram + score tables.

    Supports both A→B and B→A attention directions.

    Parameters
    ----------
    matrix_ab : A-attends-to-B normalised matrix (rows = A, cols = B)
    raw_ab    : raw (un-normalised) A→B weights
    matrix_ba : B-attends-to-A normalised matrix (rows = B, cols = A)
    raw_ba    : raw B→A weights
    direction : ``"a_to_b"`` or ``"b_to_a"`` — which direction to visualise
    """
    if raw_ab is None:
        raw_ab = matrix_ab
    if matrix_ba is None:
        matrix_ba = [[0.0] * len(words_a) for _ in words_b]
    if raw_ba is None:
        raw_ba = matrix_ba

    if direction == "b_to_a":
        active_mat = matrix_ba
        active_raw = raw_ba
    else:
        active_mat = matrix_ab
        active_raw = raw_ab

    n_a = len(words_a)
    n_b = len(words_b)

    mat = np.array(active_mat)
    global_max = float(mat.max()) if mat.size else 1.0
    if global_max == 0:
        global_max = 1.0

    row_h     = 36
    pad_top   = 70
    pad_bot   = 30
    score_w   = 65
    col_left  = 200 + score_w
    col_right = 540 + score_w
    svg_w     = 760 + score_w * 2
    n_rows_max = max(n_a, n_b)
    svg_h     = pad_top + n_rows_max * row_h + pad_bot

    has_sel = sel_side is not None and sel_idx is not None

    def y_pos(idx, total):
        block_h = total * row_h
        offset = (svg_h - pad_top - pad_bot - block_h) / 2
        return pad_top + offset + idx * row_h + row_h / 2

    def _entity_fill(word: str, default: str) -> str:
        if not show_entity_colors:
            return default
        et = _token_entity_type(word)
        if et:
            return _ENTITY_COLORS[et][0]
        return default

    # ── direction label ──
    dir_label = "A → B" if direction == "a_to_b" else "B → A"

    svg_parts = [
        f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="#1a1a2e"/>',
        f'<text x="{svg_w // 2}" y="25" text-anchor="middle" font-size="16" '
        f'font-family="sans-serif" fill="#aaa">Attention direction: {dir_label}</text>',
        f'<text x="{col_left - 5}" y="50" text-anchor="end" font-size="13" '
        f'font-family="sans-serif" fill="#8888cc">String A</text>',
        f'<text x="{col_right + 5}" y="50" text-anchor="start" font-size="13" '
        f'font-family="sans-serif" fill="#88cc88">String B</text>',
    ]

    # ── lines ──
    for i in range(n_a):
        for j in range(n_b):
            if direction == "a_to_b":
                score = active_mat[i][j]
            else:
                score = active_mat[j][i]

            ya = y_pos(i, n_a)
            yb = y_pos(j, n_b)

            if has_sel:
                is_active = ((sel_side == "a" and i == sel_idx)
                             or (sel_side == "b" and j == sel_idx))
                if is_active:
                    raw_v = score / global_max
                    opacity = max(raw_v ** 0.5 * 0.95, 0.18)
                    color, sw = "#ff9944", "3.5"
                else:
                    opacity, color, sw = 0.03, "#4a90d9", "1"
            else:
                raw_v = score / global_max
                opacity = max(raw_v ** 0.5 * 0.9, 0.04)
                color, sw = "#4a90d9", "2"

            svg_parts.append(
                f'<line x1="{col_left + 10}" y1="{ya:.1f}" '
                f'x2="{col_right - 10}" y2="{yb:.1f}" '
                f'stroke="{color}" stroke-width="{sw}" opacity="{opacity:.4f}" />'
            )

    # ── word labels A (left) ──
    for i, w in enumerate(words_a):
        ya = y_pos(i, n_a)
        esc = html.escape(w)
        is_sel = has_sel and sel_side == "a" and i == sel_idx
        fill = "#ffffff" if is_sel else _entity_fill(w, "#e0e0e0")
        fw = "bold" if is_sel else "normal"

        if direction == "a_to_b":
            row_score = sum(active_mat[i])
        else:
            row_score = sum(active_mat[j][i] for j in range(n_b))
        bg_opacity = min(row_score / (global_max * max(n_b, 1)) * 2, 0.6)

        svg_parts.append(
            f'<rect x="{col_left - 170}" y="{ya - 14}" width="175" height="28" '
            f'rx="4" fill="rgba(74,144,217,{bg_opacity:.2f})" />'
            f'<text x="{col_left - 5}" y="{ya + 5}" text-anchor="end" '
            f'font-size="14" font-family="monospace" fill="{fill}" '
            f'font-weight="{fw}">{esc}</text>'
        )

        if has_sel and sel_side == "a" and i == sel_idx:
            for j in range(n_b):
                if direction == "a_to_b":
                    s = active_mat[i][j]
                else:
                    s = active_mat[j][i]
                if s > 0.01:
                    yb = y_pos(j, n_b)
                    svg_parts.append(
                        f'<text x="{col_right + 155}" y="{yb + 5}" '
                        f'text-anchor="end" font-size="11" font-family="monospace" '
                        f'fill="#ff9944">{s:.1%}</text>'
                    )

    # ── word labels B (right) ──
    for j, w in enumerate(words_b):
        yb = y_pos(j, n_b)
        esc = html.escape(w)
        is_sel = has_sel and sel_side == "b" and j == sel_idx
        fill = "#ffffff" if is_sel else _entity_fill(w, "#e0e0e0")
        fw = "bold" if is_sel else "normal"

        if direction == "a_to_b":
            col_score = sum(active_mat[i][j] for i in range(n_a))
        else:
            col_score = sum(active_mat[j])
        bg_opacity = min(col_score / (global_max * max(n_a, 1)) * 2, 0.6)

        svg_parts.append(
            f'<rect x="{col_right}" y="{yb - 14}" width="175" height="28" '
            f'rx="4" fill="rgba(80,184,72,{bg_opacity:.2f})" />'
            f'<text x="{col_right + 5}" y="{yb + 5}" text-anchor="start" '
            f'font-size="14" font-family="monospace" fill="{fill}" '
            f'font-weight="{fw}">{esc}</text>'
        )

        if has_sel and sel_side == "b" and j == sel_idx:
            for i in range(n_a):
                if direction == "a_to_b":
                    s = active_mat[i][j]
                else:
                    s = active_mat[j][i]
                if s > 0.01:
                    ya = y_pos(i, n_a)
                    svg_parts.append(
                        f'<text x="{col_left - 175}" y="{ya + 5}" '
                        f'text-anchor="start" font-size="11" font-family="monospace" '
                        f'fill="#ff9944">{s:.1%}</text>'
                    )

    svg_parts.append("</svg>")

    # ── legend ──
    if show_entity_colors:
        legend = (
            '<div style="display:flex;gap:20px;margin:8px 0;font-size:13px;color:#ccc">'
            '<span><span style="color:#4a90d9">&#9632;</span> name</span>'
            '<span><span style="color:#50b848">&#9632;</span> date</span>'
            '<span><span style="color:#e8a838">&#9632;</span> number</span>'
            '</div>'
        )
    else:
        legend = ""

    # ── summary tables ──
    _raw = np.array(active_raw)
    summary_html = ['<div style="display:flex;gap:30px;margin-top:12px">']

    # String A summary
    if direction == "a_to_b":
        row_sums = _raw.sum(axis=1)
    else:
        row_sums = np.array([_raw[:, i].sum() for i in range(n_a)])
    row_max = float(row_sums.max()) if row_sums.size else 1.0
    if row_max == 0:
        row_max = 1.0

    summary_html.append(
        '<div style="flex:1"><table style="width:100%;border-collapse:collapse;color:#ccc;font-size:13px">'
        '<tr><th colspan="3" style="text-align:left;padding:6px;color:#8888cc">'
        'Raw attention given by each word in String A</th></tr>'
    )
    a_ranked = sorted(range(n_a), key=lambda i: row_sums[i], reverse=True)
    for i in a_ranked:
        total = float(row_sums[i])
        bar_w = int(total / row_max * 200)
        esc_a = html.escape(words_a[i])
        fill = _entity_fill(words_a[i], "#ccc")
        summary_html.append(
            f'<tr style="border-bottom:1px solid #222">'
            f'<td style="padding:5px 10px;color:{fill}">{esc_a}</td>'
            f'<td style="padding:5px 10px">'
            f'<div style="background:#333;border-radius:3px;height:16px;width:200px">'
            f'<div style="background:#4a90d9;border-radius:3px;height:16px;'
            f'width:{bar_w}px"></div></div></td>'
            f'<td style="padding:5px 10px;text-align:right;color:#eee;'
            f'font-weight:bold">{total:.4f}</td></tr>'
        )
    summary_html.append('</table></div>')

    # String B summary
    if direction == "a_to_b":
        col_sums = _raw.sum(axis=0)
    else:
        col_sums = np.array([_raw[j].sum() for j in range(n_b)])
    col_max = float(col_sums.max()) if col_sums.size else 1.0
    if col_max == 0:
        col_max = 1.0

    summary_html.append(
        '<div style="flex:1"><table style="width:100%;border-collapse:collapse;color:#ccc;font-size:13px">'
        '<tr><th colspan="3" style="text-align:left;padding:6px;color:#88cc88">'
        'Raw attention received by each word in String B</th></tr>'
    )
    b_ranked = sorted(range(n_b), key=lambda j: col_sums[j], reverse=True)
    for j in b_ranked:
        total = float(col_sums[j])
        bar_w = int(total / col_max * 200)
        esc_b = html.escape(words_b[j])
        fill = _entity_fill(words_b[j], "#ccc")
        summary_html.append(
            f'<tr style="border-bottom:1px solid #222">'
            f'<td style="padding:5px 10px;color:{fill}">{esc_b}</td>'
            f'<td style="padding:5px 10px">'
            f'<div style="background:#333;border-radius:3px;height:16px;width:200px">'
            f'<div style="background:#50b848;border-radius:3px;height:16px;'
            f'width:{bar_w}px"></div></div></td>'
            f'<td style="padding:5px 10px;text-align:right;color:#eee;'
            f'font-weight:bold">{total:.4f}</td></tr>'
        )
    summary_html.append('</table></div>')
    summary_html.append('</div>')

    return (
        '<div style="background:#1a1a2e;padding:15px;border-radius:8px">'
        + legend
        + "\n".join(svg_parts)
        + "\n".join(summary_html)
        + '</div>'
    )


def build_app():
    """Build and return the Gradio application."""
    import gradio as gr

    info = model_info()
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]

    layer_choices = ["all"] + [str(i) for i in range(n_layers)]
    head_choices = ["all"] + [str(i) for i in range(n_heads)]

    with gr.Blocks(
        title="BERT Attention Explorer",
        theme=gr.themes.Base(primary_hue="blue"),
    ) as app:
        gr.Markdown("## BERT Bidirectional Attention Explorer")
        gr.Markdown(
            "Visualise cross-attention between a query (String A) and document text "
            "(String B).  BERT sees **both directions** — no causal mask."
        )

        with gr.Row():
            string_a = gr.Textbox(
                label="String A (query)",
                value="when was it filed?",
                lines=1,
            )
            string_b = gr.Textbox(
                label="String B (document)",
                value="FILED: NEW YORK COUNTY CLERK 03/22/2021 03:01 PM",
                lines=2,
            )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=[DEFAULT_MODEL, LEGAL_MODEL],
                value=DEFAULT_MODEL,
                label="Model",
            )
            layer_dd = gr.Dropdown(choices=layer_choices, value="all", label="Layer")
            head_dd = gr.Dropdown(choices=head_choices, value="all", label="Head")
            dir_dd = gr.Dropdown(
                choices=["a_to_b", "b_to_a"],
                value="a_to_b",
                label="Direction",
            )
            entity_cb = gr.Checkbox(value=True, label="Entity colours")

        run_btn = gr.Button("Compute Attention", variant="primary")

        with gr.Row():
            sel_a = gr.Dropdown(choices=[], label="Select word from A", interactive=True)
            sel_b = gr.Dropdown(choices=[], label="Select word from B", interactive=True)

        diagram = gr.HTML(label="Attention Diagram")

        # -- State --
        st_words_a = gr.State([])
        st_words_b = gr.State([])
        st_mat_ab = gr.State([])
        st_raw_ab = gr.State([])
        st_mat_ba = gr.State([])
        st_raw_ba = gr.State([])

        def on_compute(sa, sb, mdl, lay, hd, dirn, ecol):
            wa, wb, norm_ab, raw_ab = compute_cross_attention(sa, sb, mdl, lay, hd)
            wb2, wa2, norm_ba, raw_ba = compute_cross_attention(sb, sa, mdl, lay, hd)

            dd_a = gr.update(choices=[f"{i}: {w}" for i, w in enumerate(wa)], value=None)
            dd_b = gr.update(choices=[f"{i}: {w}" for i, w in enumerate(wb)], value=None)

            html_out = render_diagram(
                wa, wb, norm_ab, raw_ab, norm_ba, raw_ba,
                direction=dirn, show_entity_colors=ecol,
            )
            return wa, wb, norm_ab, raw_ab, norm_ba, raw_ba, dd_a, dd_b, html_out

        def on_select_a(val, wa, wb, mab, rab, mba, rba, dirn, ecol):
            if val is None:
                return render_diagram(wa, wb, mab, rab, mba, rba, direction=dirn, show_entity_colors=ecol)
            idx = int(val.split(":")[0])
            return render_diagram(wa, wb, mab, rab, mba, rba, sel_side="a", sel_idx=idx,
                                  direction=dirn, show_entity_colors=ecol)

        def on_select_b(val, wa, wb, mab, rab, mba, rba, dirn, ecol):
            if val is None:
                return render_diagram(wa, wb, mab, rab, mba, rba, direction=dirn, show_entity_colors=ecol)
            idx = int(val.split(":")[0])
            return render_diagram(wa, wb, mab, rab, mba, rba, sel_side="b", sel_idx=idx,
                                  direction=dirn, show_entity_colors=ecol)

        run_btn.click(
            fn=on_compute,
            inputs=[string_a, string_b, model_dd, layer_dd, head_dd, dir_dd, entity_cb],
            outputs=[st_words_a, st_words_b, st_mat_ab, st_raw_ab,
                     st_mat_ba, st_raw_ba, sel_a, sel_b, diagram],
        )

        sel_a.change(
            fn=on_select_a,
            inputs=[sel_a, st_words_a, st_words_b, st_mat_ab, st_raw_ab,
                    st_mat_ba, st_raw_ba, dir_dd, entity_cb],
            outputs=[diagram],
        )

        sel_b.change(
            fn=on_select_b,
            inputs=[sel_b, st_words_a, st_words_b, st_mat_ab, st_raw_ab,
                    st_mat_ba, st_raw_ba, dir_dd, entity_cb],
            outputs=[diagram],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_port=7861)
