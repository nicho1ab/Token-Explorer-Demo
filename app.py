"""
Token Explorer for Educators - Streamlit Application
Performance + Accessibility build:
- Cached Kaleido/Chromium configuration for Plotly image export
- Heavy renders (PNG, PDF, Word Cloud) gated behind buttons
- Cached asset generation keyed by prompt+params hash
- High-contrast theme toggle with dynamic CSS
- Global font-size control via CSS on <html>
- ARIA roles and labels on custom HTML (glossary items, probability badges)
- Human vs AI Visualization (grouped bar)
- Confidence Tracking: Continue One Token loop

Run: streamlit run app.py
"""

import os
import shutil
import random
import math
import hashlib
from datetime import datetime
from collections import Counter
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# ---------------------------------------------------------------------
# Cached Kaleido/Chromium configuration for Streamlit Cloud
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def configure_kaleido():
    env_path = os.environ.get("PLOTLY_CHROME_PATH")
    candidates = [
        env_path,
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                pio.kaleido.scope.chromium_path = c
                return c
            except Exception:
                continue
    # Optional fallback to plotly_get_chrome if available
    try:
        import subprocess
        subprocess.run(["plotly_get_chrome"], check=True)
        c2 = shutil.which("google-chrome") or shutil.which("chromium") or "/usr/bin/google-chrome"
        if c2 and os.path.exists(c2):
            pio.kaleido.scope.chromium_path = c2
            return c2
    except Exception:
        pass
    return None

_ = configure_kaleido()

# ---------------------------------------------------------------------
# Lazy imports for heavy libs to reduce cold start
# ---------------------------------------------------------------------
def _lazy_reportlab():
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle
    return LETTER, colors, rl_canvas, inch, Table, TableStyle

def _lazy_wordcloud():
    from wordcloud import WordCloud
    from PIL import Image
    return WordCloud

# ---------------------------------------------------------------------
# Page config and base CSS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Token Explorer for Educators",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base CSS; dynamic overrides are injected later
st.markdown("""
<style>
    .stButton>button { min-height: 44px; min-width: 44px; font-size: 16px; }
    .probability-high { background-color: #28A745; color: #FFFFFF; padding: 8px; border-radius: 6px; }
    .probability-medium { background-color: #17A2B8; color: #FFFFFF; padding: 8px; border-radius: 6px; }
    .probability-low { background-color: #FFC107; color: #111111; padding: 8px; border-radius: 6px; }
    .probability-verylow { background-color: #6C757D; color: #FFFFFF; padding: 8px; border-radius: 6px; }
    .visually-hidden { position:absolute !important; height:1px; width:1px; overflow:hidden; clip:rect(1px,1px,1px,1px); white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------
GLOSSARY = {
    "Token": {"simple": "A piece of text the model sees: a word, part of a word, or punctuation.",
              "detailed": "Tokens are the basic units. Some words split into subwords for efficiency."},
    "Probability": {"simple": "How likely the model thinks a token should come next.",
                    "detailed": "A normalized score across all possible next tokens."},
    "Temperature": {"simple": "Controls creativity vs predictability.",
                    "detailed": "Scales logits before softmax. Low = deterministic, high = diverse."},
    "Top-k": {"simple": "Keep only the k most likely tokens.",
              "detailed": "Fixed-size shortlist to reduce noise."},
    "Top-p (Nucleus)": {"simple": "Keep smallest set of tokens summing to p.",
                        "detailed": "Adaptive shortlist based on uncertainty."},
    "Perplexity": {"simple": "Lower is better. Measures confusion.",
                   "detailed": "Equivalent number of equally likely options."},
    "Entropy": {"simple": "Higher means more uncertainty.",
                "detailed": "Shannon entropy of the next-token distribution."}
}

EXAMPLE_PROMPTS = {
    "Famous Quotes": [
        "To be or not to be,",
        "I have a dream that one day",
        "Ask not what your country can do for you",
        "In the beginning was the",
        "Four score and seven years ago"
    ],
    "Story Starters": [
        "Once upon a time in a",
        "It was a dark and stormy",
        "The detective walked into the room and",
        "Long ago in a galaxy far",
        "The treasure map showed"
    ],
    "Science Facts": [
        "Water boils at",
        "The Earth revolves around",
        "Photosynthesis is the process by which",
        "DNA stands for",
        "The speed of light is"
    ],
    "Simple Sentences": [
        "The cat sat on the",
        "She walked to the",
        "Today is a beautiful",
        "The student studied for the",
        "My favorite color is"
    ],
    "Math & Logic": [
        "Two plus two equals",
        "If it rains then",
        "The square root of 16 is",
        "All mammals have",
        "When water freezes it becomes"
    ]
}

MODELS = {
    "GPT-2 (English)": {"vocab_size": 50257, "languages": ["English"],
                        "description": "General-purpose English model",
                        "best_for": "Story writing, general predictions"},
    "BERT Base (English)": {"vocab_size": 30522, "languages": ["English"],
                            "description": "Mask prediction, strong context understanding",
                            "best_for": "Fill-in-the-blank"},
    "BERT Multilingual": {"vocab_size": 119547,
                          "languages": ["English","Spanish","French","German","Chinese","Arabic","Hindi","104 total"],
                          "description": "Supports 104 languages",
                          "best_for": "Multilingual text"},
    "GPT-2 Spanish": {"vocab_size": 50257, "languages": ["Spanish"],
                      "description": "Spanish generation",
                      "best_for": "Spanish text generation"},
    "DistilGPT-2 (Fast)": {"vocab_size": 50257, "languages": ["English"],
                           "description": "Smaller, faster GPT-2",
                           "best_for": "Quick demos"}
}

ACTIVITIES = {
    "Predict the Next Word Game": {
        "grade_level": "3-8",
        "duration": "15-20 minutes",
        "description": "Students guess the next word, compare to AI predictions.",
        "steps": [
            "Display a sentence with the last word hidden",
            "Have students write their predictions",
            "Reveal AI's top predictions with probabilities",
            "Discuss why certain words are more likely",
            "Change temperature and compare shifts"
        ],
        "learning_goals": [
            "Probability and prediction",
            "Pattern recognition",
            "Intro to AI decision-making"
        ]
    },
    "Temperature Experiment": {
        "grade_level": "6-12",
        "duration": "25-30 minutes",
        "description": "Explore how temperature affects creativity and coherence.",
        "steps": [
            "Start with the same prompt for all students",
            "Generate predictions at a low temperature (e.g., 0.2)",
            "Generate predictions at a high temperature (e.g., 1.5)",
            "Compare and discuss differences",
            "Students chart variety vs. coherence"
        ],
        "learning_goals": [
            "Understanding parameters in AI systems",
            "Balancing creativity and accuracy",
            "Data analysis and comparison"
        ]
    }
}

# ---------------------------------------------------------------------
# Probability and metrics
# ---------------------------------------------------------------------
def generate_probabilities(prompt, model_name, temperature, top_k, top_p):
    context_predictions = {
        "The cat sat on the": {"mat": 0.35, "chair": 0.20, "floor": 0.15, "table": 0.12,
                               "sofa": 0.08, "bed": 0.05, "couch": 0.03, "roof": 0.02},
        "Once upon a time in a": {"kingdom": 0.40, "land": 0.25, "forest": 0.15, "village": 0.10,
                                  "castle": 0.05, "city": 0.03, "galaxy": 0.02},
        "Water boils at": {"100": 0.70, "212": 0.15, "boiling": 0.05, "high": 0.04,
                           "sea": 0.03, "room": 0.02, "atmospheric": 0.01},
        "To be or not to be,": {"that": 0.90, "this": 0.03, "whether": 0.02, "it": 0.02,
                                "what": 0.01, "which": 0.01, "the": 0.01},
        "The Earth revolves around": {"the": 0.85, "Sun": 0.08, "its": 0.03, "a": 0.02,
                                      "our": 0.01, "itself": 0.01}
    }
    base = None
    for k in context_predictions:
        if k.lower() in prompt.lower():
            base = context_predictions[k].copy()
            break
    if base is None:
        default_tokens = ["the", "a", "and", "is", "to", "of", "in", "it", "for", "on"]
        base = {t: random.uniform(0.05, 0.25) for t in default_tokens}
        s = sum(base.values())
        base = {k: v/s for k, v in base.items()}

    # temperature scaling
    if temperature > 0:
        logits = {k: math.log(v) / max(temperature, 1e-6) for k, v in base.items()}
        m = max(logits.values())
        exps = {k: math.exp(v - m) for k, v in logits.items()}
        s = sum(exps.values())
        base = {k: v/s for k, v in exps.items()}

    # top-k
    if top_k > 0:
        items = sorted(base.items(), key=lambda x: x[1], reverse=True)[:top_k]
        base = dict(items)
        s = sum(base.values())
        base = {k: v/s for k, v in base.items()}

    # top-p
    if top_p < 1.0:
        items = sorted(base.items(), key=lambda x: x[1], reverse=True)
        csum = 0.0
        nucleus = []
        for tok, p in items:
            nucleus.append((tok, p))
            csum += p
            if csum >= top_p:
                break
        base = dict(nucleus)
        s = sum(base.values())
        base = {k: v/s for k, v in base.items()}

    return dict(sorted(base.items(), key=lambda x: x[1], reverse=True))

def calculate_entropy(probabilities):
    e = 0.0
    for p in probabilities.values():
        if p > 0:
            e -= p * math.log2(p)
    return e

def calculate_perplexity(entropy):
    return 2 ** entropy

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
def create_probability_chart(predictions):
    tokens = list(predictions.keys())[:10]
    probs = [predictions[t] * 100 for t in tokens]
    colors_list = []
    for p in probs:
        if p > 50: colors_list.append('#28A745')
        elif p > 20: colors_list.append('#17A2B8')
        elif p > 5: colors_list.append('#FFC107')
        else: colors_list.append('#6C757D')
    fig = go.Figure(data=[go.Bar(
        y=tokens, x=probs, orientation='h',
        marker=dict(color=colors_list),
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside'
    )])
    fig.update_layout(
        title="Top 10 Token Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Token",
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_entropy_chart(entropy_values):
    if not entropy_values:
        fig = go.Figure()
        fig.update_layout(title="Entropy Over Token Sequence",
                          xaxis_title="Step", yaxis_title="Entropy (bits)")
        return fig
    fig = go.Figure(data=[go.Scatter(
        x=list(range(1, len(entropy_values)+1)),
        y=entropy_values,
        mode='lines+markers',
        marker=dict(size=8, color='#0066CC'),
        line=dict(width=2, color='#0066CC')
    )])
    fig.update_layout(
        title="Entropy Over Token Sequence",
        xaxis_title="Step",
        yaxis_title="Entropy (bits)",
        height=400
    )
    return fig

# ---------------------------------------------------------------------
# Hashing and cached heavy renders
# ---------------------------------------------------------------------
def _hash_key(prompt: str, model: str, temperature: float, top_k: int, top_p: float, predictions: dict) -> str:
    base = f"{prompt}|{model}|{temperature:.3f}|{top_k}|{top_p:.3f}|" + ";".join(f"{k}:{predictions[k]:.8f}" for k in sorted(predictions.keys()))
    return hashlib.sha1(base.encode()).hexdigest()

@st.cache_data(show_spinner=False)
def render_chart_png_cached(fig_dict: dict, width=1000, height=600, scale=1) -> bytes:
    fig = go.Figure(fig_dict)
    return pio.to_image(fig, format="png", width=width, height=height, scale=scale)

@st.cache_data(show_spinner=False)
def render_wordcloud_png_cached(predictions: dict, width=800, height=400) -> bytes:
    WordCloud = _lazy_wordcloud()
    wc = WordCloud(width=width, height=height, background_color='white')
    wc.generate_from_frequencies(predictions)
    from PIL import Image
    img = Image.fromarray(wc.to_array())
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio.read()

@st.cache_data(show_spinner=False)
def render_prediction_pdf_cached(prompt_text: str, params: dict, metrics: dict, predictions: dict, fig_dict: dict | None) -> bytes:
    LETTER, colors, rl_canvas, inch, Table, TableStyle = _lazy_reportlab()
    # Recreate chart PNG only if fig_dict provided
    chart_png = None
    if fig_dict:
        fig = go.Figure(fig_dict)
        chart_png = pio.to_image(fig, format="png", width=1000, height=600, scale=1)

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    x = margin
    y = height - margin

    # Helpers
    def _draw_wrapped_text(text, x_, y_, max_width, line_height=14, font_name="Helvetica", font_size=10):
        c.setFont(font_name, font_size)
        words = text.split()
        line = ""
        while words:
            w = words[0]
            test = f"{line} {w}".strip()
            if c.stringWidth(test, font_name, font_size) <= max_width:
                line = test; words.pop(0)
            else:
                c.drawString(x_, y_, line); y_ -= line_height; line = ""
        if line:
            c.drawString(x_, y_, line); y_ -= line_height
        return y_

    def _build_top_tokens_table_data(preds: dict, top_n: int = 10):
        rows = [["Rank", "Token", "Probability"]]
        for i, (tok, p) in enumerate(list(preds.items())[:top_n], start=1):
            rows.append([i, tok, f"{p*100:.2f}%"])
        return rows

    # Header
    c.setTitle("Token Explorer Report")
    c.setFont("Helvetica-Bold", 16); c.drawString(x, y, "Token Explorer for Educators — Prediction Report")
    y -= 18
    c.setFont("Helvetica", 10); c.setFillColor(colors.grey)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFillColor(colors.black); y -= 22
    c.line(x, y, width - margin, y); y -= 18

    # Prompt
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Input Prompt:"); y -= 16
    y = _draw_wrapped_text(prompt_text or "(none)", x, y, max_width=width - 2*margin, font_size=11)

    # Params
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Parameters:"); y -= 16
    c.setFont("Helvetica", 11)
    for line in [
        f"Temperature: {params.get('temperature')}",
        f"Top-k: {params.get('top_k')}",
        f"Top-p: {params.get('top_p')}",
        f"Model: {params.get('model_name')}",
    ]:
        c.drawString(x, y, line); y -= 14

    # Metrics
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Metrics:"); y -= 16
    c.setFont("Helvetica", 11)
    m_lines = [
        f"Entropy: {metrics.get('entropy'):.2f} bits" if metrics.get("entropy") is not None else "Entropy: n/a",
        f"Perplexity: {metrics.get('perplexity'):.2f}" if metrics.get("perplexity") is not None else "Perplexity: n/a",
        f"Top Token Probability: {max(predictions.values())*100:.1f}%" if predictions else "Top Token Probability: n/a",
    ]
    for line in m_lines:
        c.drawString(x, y, line); y -= 14

    # Table
    y -= 10
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Top-10 Tokens:"); y -= 16
    table_data = _build_top_tokens_table_data(predictions or {}, 10)
    tbl = Table(table_data, colWidths=[0.8*inch, 3.2*inch, 1.3*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E7F3FF")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#000000")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#B0B0B0")),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (-1,-1), 10),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    needed_height = 14 * len(table_data)
    if y - needed_height < margin + 220:
        c.showPage(); y = height - margin
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Top-10 Tokens (cont'd):"); y -= 18
    tbl.wrapOn(c, width - 2*margin, y)
    tbl.drawOn(c, x, y - (14 * len(table_data)))
    y -= (14 * len(table_data) + 18)

    # Chart
    if chart_png:
        if y < margin + 220:
            c.showPage(); y = height - margin
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Probability Chart:"); y -= 12
        img_width = width - 2*margin; img_height = img_width * 0.55
        c.drawImage(BytesIO(chart_png), x, y - img_height, width=img_width, height=img_height,
                    preserveAspectRatio=True, mask='auto')
        y -= (img_height + 6)

    c.showPage(); c.save(); buf.seek(0)
    return buf.read()

@st.cache_data(show_spinner=False)
def render_handout_pdf_cached(activity_title: str, activity: dict) -> bytes:
    LETTER, colors, rl_canvas, inch, Table, TableStyle = _lazy_reportlab()
    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    x = margin
    y = height - margin

    def _draw_wrapped_text(text, x_, y_, max_width, line_height=14, font_name="Helvetica", font_size=10):
        c.setFont(font_name, font_size)
        words = text.split()
        line = ""
        while words:
            w = words[0]
            test = f"{line} {w}".strip()
            if c.stringWidth(test, font_name, font_size) <= max_width:
                line = test; words.pop(0)
            else:
                c.drawString(x_, y_, line); y_ -= line_height; line = ""
        if line:
            c.drawString(x_, y_, line); y_ -= line_height
        return y_

    c.setTitle(f"{activity_title} - Handout")
    c.setFont("Helvetica-Bold", 18); c.drawString(x, y, activity_title)
    y -= 20
    c.setFont("Helvetica", 10); c.setFillColor(colors.grey)
    c.drawString(x, y, f"Printable Activity Handout • Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFillColor(colors.black); y -= 14
    c.line(x, y, width - margin, y); y -= 18

    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Grade Level:"); c.setFont("Helvetica", 12)
    c.drawString(x + 90, y, activity.get("grade_level","N/A")); y -= 16
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Duration:"); c.setFont("Helvetica", 12)
    c.drawString(x + 90, y, activity.get("duration","N/A")); y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Description:")
    y -= 16
    y = _draw_wrapped_text(activity.get("description",""), x, y, max_width=width - 2*margin, font_size=11)

    y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Steps:")
    y -= 16
    c.setFont("Helvetica", 11)
    for idx, step in enumerate(activity.get("steps", []), start=1):
        y = _draw_wrapped_text(f"{idx}. {step}", x, y, max_width=width - 2*margin, font_size=11)
        y -= 2
        if y < margin + 120:
            c.showPage(); y = height - margin
            c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Steps (cont'd):")
            y -= 16; c.setFont("Helvetica", 11)

    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Learning Goals:")
    y -= 16; c.setFont("Helvetica", 11)
    for goal in activity.get("learning_goals", []):
        y = _draw_wrapped_text(f"• {goal}", x, y, max_width=width - 2*margin, font_size=11)
        y -= 2
        if y < margin + 80:
            c.showPage(); y = height - margin
            c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Learning Goals (cont'd):")
            y -= 16; c.setFont("Helvetica", 11)

    y = max(y, margin + 40)
    c.setFont("Helvetica-Oblique", 9); c.setFillColor(colors.grey)
    c.drawString(x, margin, "Token Explorer for Educators • Handout")
    c.setFillColor(colors.black)

    c.showPage(); c.save(); buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------
def _ensure_state_defaults():
    st.session_state.setdefault('tutorial_shown', False)
    st.session_state.setdefault('show_tutorial', False)
    st.session_state.setdefault('show_glossary', False)
    st.session_state.setdefault('poll_mode', False)
    st.session_state.setdefault('student_predictions', [])

    # Accessibility
    st.session_state.setdefault('high_contrast', False)
    st.session_state.setdefault('font_size', 'Medium')  # Small / Medium / Large

    # Params
    st.session_state.setdefault('temperature', 1.0)
    st.session_state.setdefault('top_k', 50)
    st.session_state.setdefault('top_p', 0.9)

    # Current prediction context
    st.session_state.setdefault('input_text', '')
    st.session_state.setdefault('current_text', '')
    st.session_state.setdefault('current_model', list(MODELS.keys())[0])
    st.session_state.setdefault('predictions', None)
    st.session_state.setdefault('entropy', None)
    st.session_state.setdefault('perplexity', None)

    # Confidence tracking sequence
    st.session_state.setdefault('sequence_tokens', [])
    st.session_state.setdefault('sequence_entropies', [])
    st.session_state.setdefault('sequence_top1_probs', [])

_ensure_state_defaults()

# ---------------------------------------------------------------------
# Dynamic style injection
# ---------------------------------------------------------------------
def apply_dynamic_styles():
    font_map = {"Small": "14px", "Medium": "16px", "Large": "18px"}
    base_size = font_map.get(st.session_state.get("font_size", "Medium"), "16px")

    css_parts = [f"""
    <style>
      html {{ font-size: {base_size}; }}
      body, .markdown-text-container, .stMarkdown, .stText, .stRadio, .stSelectbox, .stMultiSelect,
      .stDataFrame, .stMetric, .stTextInput, .stTextArea, .stSlider, .stDownloadButton, .stButton, .stExpander,
      .stTabs, .stTab {{ font-size: 1rem; line-height: 1.5; }}
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"],
      .stButton>button, .stDownloadButton>button {{ font-size: 1rem; }}
    """]

    if st.session_state.get("high_contrast", False):
        css_parts.append("""
        html, body { background-color: #000000 !important; color: #FFFFFF !important; }
        .stApp, .block-container { background-color: #000000 !important; }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div { color: #FFFFFF !important; }
        a { color: #00E5FF !important; text-decoration: underline; }
        .stButton>button, .stDownloadButton>button {
            background-color: #FFFFFF !important; color: #000000 !important; border: 2px solid #FFFFFF !important;
        }
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox div[role="combobox"] {
            background-color: #111111 !important; color: #FFFFFF !important; border: 1px solid #FFFFFF !important;
        }
        .stDataFrame, .dataframe, .stTable { filter: invert(1) hue-rotate(180deg); }
        .probability-high { background-color: #00A65A !important; color: #FFFFFF !important; }
        .probability-medium { background-color: #148EA1 !important; color: #FFFFFF !important; }
        .probability-low { background-color: #C19A00 !important; color: #111111 !important; }
        .probability-verylow { background-color: #888888 !important; color: #FFFFFF !important; }
        """)
    css_parts.append("</style>")
    st.markdown("\n".join(css_parts), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Accessible HTML helpers
# ---------------------------------------------------------------------
def render_probability_badge(rank: int, token: str, prob: float) -> str:
    pct = prob * 100
    if pct > 50: cls = "probability-high"
    elif pct > 20: cls = "probability-medium"
    elif pct > 5: cls = "probability-low"
    else: cls = "probability-verylow"
    label = f"Rank {rank}. Token {token}. Probability {pct:.1f} percent."
    return (
        f'<div class="{cls}" role="note" aria-label="{label}" aria-live="polite">'
        f'#{rank}: <strong>{token}</strong> — {pct:.1f}%'
        f'</div>'
    )

def render_glossary_item(term: str, simple: str, detailed: str) -> str:
    term_id = f"glossary-{term.replace(' ', '-').lower()}"
    return f"""
    <section role="listitem" aria-labelledby="{term_id}">
      <h4 id="{term_id}" role="heading" aria-level="4">{term}</h4>
      <p role="note" aria-label="Simple definition for {term}">{simple}</p>
      <p role="note" aria-label="Detailed definition for {term}">{detailed}</p>
    </section>
    """

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    st.title("🎓 Token Explorer for Educators")
    st.markdown("### Visualize token-by-token prediction, confidence, and classroom activities")

    if not st.session_state.tutorial_shown:
        with st.expander("👋 Quick Start", expanded=True):
            st.markdown("""
1) Enter text or load an example  
2) Choose a model  
3) Set temperature / top-k / top-p  
4) Generate predictions  
5) Use **Continue One Token** to step and track entropy  
6) Use **Class Poll Mode** for Human vs AI visualization  
7) Print **Activity Handouts** for the classroom  
8) Exports render on demand to keep the app fast
            """)
            if st.button("Got it"):
                st.session_state.tutorial_shown = True
                st.rerun()

    # Top controls
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([2,2,2,1,1])
    with top_c1:
        if st.button("📖 Glossary"):
            st.session_state.show_glossary = not st.session_state.show_glossary
    with top_c2:
        if st.button("❓ Help"):
            st.session_state.show_tutorial = not st.session_state.show_tutorial
    with top_c3:
        st.session_state.poll_mode = st.checkbox("📊 Class Poll Mode", value=st.session_state.poll_mode)
    with top_c4:
        st.session_state.high_contrast = st.checkbox("🌓 High Contrast", value=st.session_state.high_contrast)
    with top_c5:
        st.session_state.font_size = st.selectbox("Font", ["Small","Medium","Large"],
                                                  index=["Small","Medium","Large"].index(st.session_state.font_size),
                                                  label_visibility="collapsed")

    apply_dynamic_styles()
    st.markdown("---")

    if st.session_state.show_glossary:
        with st.expander("📖 Glossary", expanded=True):
            st.markdown('<div role="list" aria-label="AI glossary terms">', unsafe_allow_html=True)
            for term, d in GLOSSARY.items():
                st.markdown(render_glossary_item(term, d['simple'], d['detailed']), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1,2,1])

    # Left: input + models
    with col_left:
        st.markdown("### 📝 Input")
        cat = st.selectbox("Load Example", ["-- Select --"] + list(EXAMPLE_PROMPTS.keys()))
        if cat != "-- Select --":
            ex = st.selectbox("Example", EXAMPLE_PROMPTS[cat])
            if st.button("Load Example"):
                st.session_state.input_text = ex
                st.session_state.current_text = ex
        if st.button("🎲 Random Example"):
            rcat = random.choice(list(EXAMPLE_PROMPTS.keys()))
            rval = random.choice(EXAMPLE_PROMPTS[rcat])
            st.session_state.input_text = rval
            st.session_state.current_text = rval

        st.session_state.input_text = st.text_area(
            "Enter text", value=st.session_state.get('input_text', ''), height=150
        )
        if st.session_state.input_text != st.session_state.current_text:
            st.session_state.current_text = st.session_state.input_text

        st.markdown("### 🤖 Model")
        model_name = st.selectbox("Choose Model", list(MODELS.keys()),
                                  index=list(MODELS.keys()).index(st.session_state.get('current_model')))
        st.session_state.current_model = model_name
        mi = MODELS[model_name]
        st.info(f"**{model_name}**  \n📊 Vocab: {mi['vocab_size']:,}  \n🌍 Languages: {', '.join(mi['languages'][:3])}{'...' if len(mi['languages'])>3 else ''}  \n✨ Best for: {mi['best_for']}")

        compare_models = st.checkbox("🔄 Compare with another model")
        model_name_2 = None
        if compare_models:
            model_name_2 = st.selectbox("Second Model", [m for m in MODELS.keys() if m != model_name])

    # Middle: Parameters inside a form to reduce reruns
    with col_mid:
        st.markdown("### 🎚️ Parameters")
        with st.form("params_form", clear_on_submit=False):
            p1, p2, p3 = st.columns(3)
            with p1:
                conservative = st.form_submit_button("🛡️ Conservative")
            with p2:
                balanced = st.form_submit_button("⚖️ Balanced")
            with p3:
                creative = st.form_submit_button("🎨 Creative")

            temp = st.slider("🌡️ Temperature", 0.0, 2.0, st.session_state.get('temperature', 1.0), 0.1)
            top_k = st.slider("🔝 Top-k", 0, 100, st.session_state.get('top_k', 50), 5)
            top_p = st.slider("🎯 Top-p", 0.0, 1.0, st.session_state.get('top_p', 0.9), 0.05)
            apply_params = st.form_submit_button("Apply")

        if conservative:
            st.session_state.update(temperature=0.3, top_k=10, top_p=0.8)
        elif balanced:
            st.session_state.update(temperature=0.8, top_k=50, top_p=0.9)
        elif creative:
            st.session_state.update(temperature=1.5, top_k=100, top_p=0.95)
        elif apply_params:
            st.session_state.update(temperature=temp, top_k=top_k, top_p=top_p)

        temperature = st.session_state.get('temperature', 1.0)
        top_k = st.session_state.get('top_k', 50)
        top_p = st.session_state.get('top_p', 0.9)

        if temperature == 0: strategy = "🔒 Greedy"
        elif top_k > 0 and top_p < 1.0: strategy = f"🎯 Top-k ({top_k}) + Top-p ({top_p})"
        elif top_k > 0: strategy = f"🔝 Top-k ({top_k})"
        elif top_p < 1.0: strategy = f"🎯 Nucleus (Top-p={top_p})"
        else: strategy = "🌡️ Temperature sampling"
        st.info(f"**Decoding Strategy:** {strategy}")

        gen_c1, gen_c2 = st.columns([2,1])
        with gen_c1:
            if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                text = st.session_state.current_text.strip()
                if text:
                    preds = generate_probabilities(text, model_name, temperature, top_k, top_p)
                    st.session_state.predictions = preds
                    ent = calculate_entropy(preds); ppl = calculate_perplexity(ent)
                    st.session_state.entropy = ent; st.session_state.perplexity = ppl
                    # reset confidence tracking
                    st.session_state.sequence_tokens = []
                    st.session_state.sequence_entropies = []
                    st.session_state.sequence_top1_probs = []
                    if compare_models and model_name_2:
                        st.session_state.predictions_2 = generate_probabilities(text, model_name_2, temperature, top_k, top_p)
                else:
                    st.warning("Enter some text.")
        with gen_c2:
            if st.button("➡️ Continue One Token", use_container_width=True):
                if st.session_state.get('predictions'):
                    curr_preds = st.session_state.predictions
                    curr_entropy = calculate_entropy(curr_preds)
                    top_token, top_prob = next(iter(curr_preds.items()))
                    st.session_state.sequence_entropies.append(curr_entropy)
                    st.session_state.sequence_top1_probs.append(top_prob)
                    st.session_state.sequence_tokens.append(top_token)

                    new_text = (st.session_state.current_text + " " + top_token).strip()
                    st.session_state.current_text = new_text
                    st.session_state.input_text = new_text

                    new_preds = generate_probabilities(new_text, st.session_state.current_model, temperature, top_k, top_p)
                    st.session_state.predictions = new_preds
                    ent = calculate_entropy(new_preds); ppl = calculate_perplexity(ent)
                    st.session_state.entropy = ent; st.session_state.perplexity = ppl
                else:
                    st.info("Generate predictions first.")

        if st.session_state.get('predictions'):
            st.markdown("---")
            st.markdown("### 🎯 Predictions", help="Top tokens with probabilities and uncertainty metrics.")
            predictions = st.session_state.predictions

            max_prob = max(predictions.values())
            if max_prob > 0.5: st.success("✅ High confidence (>50%)")
            elif max_prob > 0.2: st.info("ℹ️ Medium confidence (20–50%)")
            else: st.warning("⚠️ Low confidence (<20%)")

            m1, m2 = st.columns(2)
            with m1: st.metric("📊 Entropy", f"{st.session_state.entropy:.2f} bits")
            with m2: st.metric("🎲 Perplexity", f"{st.session_state.perplexity:.1f}")

            st.markdown('<div role="list" aria-label="Top tokens by probability" aria-live="polite">', unsafe_allow_html=True)
            for i, (tok, p) in enumerate(list(predictions.items())[:10], 1):
                st.markdown(render_probability_badge(i, tok, p), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Visualizations")
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Probability Chart", "☁️ Word Cloud", "📈 Metrics Analysis", "📉 Confidence Tracking"
            ])

            # Shared keys for caching renders
            pred_hash = _hash_key(
                st.session_state.get('current_text',''),
                st.session_state.get('current_model',''),
                temperature, top_k, top_p, predictions
            )

            with tab1:
                fig = create_probability_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)

                # On-demand PNG render
                png_ready = st.session_state.get('chart_png_ready') == pred_hash
                if st.button("🖼️ Prepare Chart Image", key=f"prep_chart_{pred_hash}"):
                    with st.spinner("Rendering PNG…"):
                        try:
                            png_bytes = render_chart_png_cached(fig.to_dict(), width=1000, height=600, scale=1)
                            st.session_state['chart_png'] = png_bytes
                            st.session_state['chart_png_ready'] = pred_hash
                            png_ready = True
                        except Exception as e:
                            st.error(f"Chart export failed: {e}")
                if png_ready:
                    st.download_button(
                        "Download Chart (PNG)",
                        data=st.session_state['chart_png'],
                        file_name=f"token_probabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )

            with tab2:
                # On-demand Word Cloud render
                wc_ready = st.session_state.get('wc_png_ready') == pred_hash
                if st.button("☁️ Prepare Word Cloud Image", key=f"prep_wc_{pred_hash}"):
                    with st.spinner("Generating Word Cloud…"):
                        try:
                            wc_png = render_wordcloud_png_cached(predictions, width=800, height=400)
                            st.session_state['wc_png'] = wc_png
                            st.session_state['wc_png_ready'] = pred_hash
                            wc_ready = True
                        except Exception as e:
                            st.error(f"Word Cloud failed: {e}")
                if wc_ready:
                    st.image(st.session_state['wc_png'], caption="Word Cloud of Token Probabilities", use_container_width=True)

            with tab3:
                st.markdown(f"""
**Confidence Metrics**
- Entropy: {st.session_state.entropy:.2f} bits
- Perplexity: {st.session_state.perplexity:.1f}
- Top Token Probability: {max(predictions.values())*100:.1f}%
                """)

                # On-demand PDF report
                pdf_ready = st.session_state.get('report_pdf_ready') == pred_hash
                if st.button("📄 Prepare Prediction Report (PDF)", key=f"prep_pdf_{pred_hash}"):
                    with st.spinner("Building PDF…"):
                        try:
                            fig = create_probability_chart(predictions)
                            pdf_bytes = render_prediction_pdf_cached(
                                prompt_text=st.session_state.get('current_text',''),
                                params={'temperature': temperature, 'top_k': top_k, 'top_p': top_p,
                                        'model_name': st.session_state.get('current_model','')},
                                metrics={'entropy': st.session_state.get('entropy'),
                                         'perplexity': st.session_state.get('perplexity')},
                                predictions=predictions,
                                fig_dict=fig.to_dict()
                            )
                            st.session_state['report_pdf'] = pdf_bytes
                            st.session_state['report_pdf_ready'] = pred_hash
                            pdf_ready = True
                        except Exception as e:
                            st.error(f"PDF export failed: {e}")
                if pdf_ready:
                    st.download_button(
                        label="Download Report (PDF)",
                        data=st.session_state['report_pdf'],
                        file_name=f"token_explorer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

            with tab4:
                st.markdown("Each **Continue One Token** records the current distribution entropy and top-1 probability before appending.")
                seq_ent = st.session_state.sequence_entropies
                seq_p1 = st.session_state.sequence_top1_probs
                seq_tok = st.session_state.sequence_tokens

                st.plotly_chart(create_entropy_chart(seq_ent), use_container_width=True)

                if seq_tok:
                    df_steps = pd.DataFrame({
                        "Step": list(range(1, len(seq_tok)+1)),
                        "Chosen Token": seq_tok,
                        "Top-1 Probability (%)": [round(p*100, 2) for p in seq_p1],
                        "Entropy (bits)": [round(e, 3) for e in seq_ent]
                    })
                    st.dataframe(df_steps, use_container_width=True)
                else:
                    st.info("No steps yet. Click **Continue One Token** to start tracking.")

                if st.button("♻️ Reset Tracking"):
                    st.session_state.sequence_tokens = []
                    st.session_state.sequence_entropies = []
                    st.session_state.sequence_top1_probs = []
                    st.success("Tracking reset.")

    # Right: activities + handout export
    with col_right:
        st.markdown("### 🏫 Classroom Activities")
        act = st.selectbox("Choose Activity", ["-- Select --"] + list(ACTIVITIES.keys()))
        if act != "-- Select --":
            a = ACTIVITIES[act]
            with st.expander(f"📋 {act}", expanded=True):
                st.markdown(f"**Grade Level**: {a['grade_level']}  \n**Duration**: {a['duration']}")
                st.markdown(f"**Description**: {a['description']}")
                st.markdown("**Steps:**")
                for i, step in enumerate(a['steps'], 1):
                    st.markdown(f"{i}. {step}")
                st.markdown("**Learning Goals:**")
                for g in a['learning_goals']:
                    st.markdown(f"- {g}")

            handout_key = hashlib.sha1(act.encode()).hexdigest()
            ready = st.session_state.get('handout_ready') == handout_key
            if st.button("📄 Prepare Handout PDF", key=f"prep_handout_{handout_key}"):
                with st.spinner("Creating handout…"):
                    try:
                        pdf_handout = render_handout_pdf_cached(act, a)
                        st.session_state['handout_pdf'] = pdf_handout
                        st.session_state['handout_ready'] = handout_key
                        ready = True
                    except Exception as e:
                        st.error(f"Handout export failed: {e}")
            if ready:
                st.download_button(
                    label="Download Handout",
                    data=st.session_state['handout_pdf'],
                    file_name=f"{act.replace(' ', '_').lower()}_handout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # Class Poll Mode
    if st.session_state.poll_mode:
        st.markdown("---")
        st.markdown("## 📊 Class Poll Mode")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 👥 Student Submissions")
            st.markdown(f"**Join Code**: POLL-{random.randint(1000, 9999)}")
            guess = st.text_input("Your prediction:", key="student_guess")
            if st.button("Submit Prediction"):
                if guess:
                    st.session_state.student_predictions.append(guess.strip().lower())
                    st.success("Submitted.")
            st.metric("Total Submissions", len(st.session_state.student_predictions))
            if st.button("🗑️ Clear All Predictions"):
                st.session_state.student_predictions = []
                st.rerun()

        with c2:
            st.markdown("### 🧠 Human vs AI")
            if st.session_state.student_predictions and st.session_state.get('predictions'):
                from collections import Counter
                counts = Counter(st.session_state.student_predictions)
                total = len(st.session_state.student_predictions)
                top5_students = counts.most_common(5)
                student_df = pd.DataFrame(
                    {"Token": [w for w, _ in top5_students],
                     "Percent": [(c/total)*100 for _, c in top5_students]}
                )

                ai_items = list(st.session_state.predictions.items())[:5]
                ai_df = pd.DataFrame(
                    {"Token": [t for t, _ in ai_items],
                     "Percent": [p*100 for _, p in ai_items]}
                )

                tokens_union = sorted(set(student_df["Token"]).union(set(ai_df["Token"])))
                combined_rows = []
                for tok in tokens_union:
                    s_val = float(student_df.loc[student_df["Token"] == tok, "Percent"].iloc[0]) if (student_df["Token"] == tok).any() else 0.0
                    a_val = float(ai_df.loc[ai_df["Token"] == tok, "Percent"].iloc[0]) if (ai_df["Token"] == tok).any() else 0.0
                    combined_rows.append({"Token": tok, "Source": "Students", "Percent": s_val})
                    combined_rows.append({"Token": tok, "Source": "AI", "Percent": a_val})

                combined_df = pd.DataFrame(combined_rows)

                st.dataframe(
                    pd.concat([
                        student_df.assign(Source="Students"),
                        ai_df.assign(Source="AI")
                    ], ignore_index=True),
                    use_container_width=True
                )

                fig_cmp = px.bar(
                    combined_df,
                    x="Token", y="Percent",
                    color="Source",
                    barmode="group",
                    title="Student vs AI Predictions",
                    text=combined_df["Percent"].round(1).astype(str) + "%"
                )
                fig_cmp.update_layout(yaxis_title="Percent (%)", xaxis_title="Token", height=450)
                st.plotly_chart(fig_cmp, use_container_width=True)

                top_student = top5_students[0][0]
                top_ai = ai_items[0][0]
                if top_student == top_ai:
                    st.success(f"✅ Agreement on '{top_student}'")
                else:
                    st.info(f"Students: '{top_student}' vs AI: '{top_ai}'")
            else:
                st.info("Need at least one student submission and an AI prediction to display the comparison.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6C757D; padding: 20px;'>
      <p><strong>Token Explorer for Educators</strong> | Fast & Accessible Edition</p>
      <p>On-demand exports and caching reduce load time</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
