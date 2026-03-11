"""
BERT bidirectional attention exploration for legal document metadata extraction.

Modules:
    attention_probe  — core BERT loader, cross-attention computation
    head_analysis    — probe 144 heads for name/date/number specialization
    extract_with_heads — use discovered heads to extract metadata
    viz_ui           — interactive Gradio visualization
"""
