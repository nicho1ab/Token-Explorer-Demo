"""
Token Explorer for Educators - Streamlit Application
Enhanced version with real PNG export via plotly.io.to_image
Deploy with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json
from datetime import datetime
import random
import math
from collections import Counter
from io import BytesIO

# PDF generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle

# Page configuration
st.set_page_config(
    page_title="Token Explorer for Educators",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and accessibility
st.markdown("""
<style>
    .stButton>button {
        min-height: 44px;
        min-width: 44px;
        font-size: 16px;
    }
    .probability-high { background-color: #28A745; color: white; padding: 8px; border-radius: 5px; }
    .probability-medium { background-color: #17A2B8; color: white; padding: 8px; border-radius: 5px; }
    .probability-low { background-color: #FFC107; color: black; padding: 8px; border-radius: 5px; }
    .probability-verylow { background-color: #6C757D; color: white; padding: 8px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Educational content and glossary
GLOSSARY = {
    "Token": {
        "simple": "A piece of text that the AI model understands - like a word, part of a word, or punctuation.",
        "detailed": "Tokens are the basic units that language models use to process text. A word like 'running' might be one token, while 'unbelievable' might be split into 'un', 'believe', and 'able'."
    },
    "Probability": {
        "simple": "How likely the AI thinks a word should come next, as a percentage.",
        "detailed": "The model calculates probabilities for thousands of possible next tokens. Higher probability means a more likely next token."
    },
    "Temperature": {
        "simple": "Controls how creative or predictable the AI is.",
        "detailed": "Temperature scales logits before softmax. Low = deterministic, high = more diverse and risky."
    },
    "Top-k": {
        "simple": "Consider only the k most likely tokens.",
        "detailed": "Restricts sampling to top k tokens to avoid long-tail noise."
    },
    "Top-p (Nucleus)": {
        "simple": "Consider the smallest set of tokens whose cumulative probability ≥ p.",
        "detailed": "Dynamically chooses the shortlist size based on uncertainty."
    },
    "Perplexity": {
        "simple": "Lower is better. Measures confusion.",
        "detailed": "Roughly equals the number of equally likely choices the model is choosing among."
    },
    "Entropy": {
        "simple": "Higher means more uncertain.",
        "detailed": "Shannon entropy in bits over the next-token distribution."
    }
}

# Example prompts by category
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

# Model configurations
MODELS = {
    "GPT-2 (English)": {
        "vocab_size": 50257,
        "languages": ["English"],
        "description": "General-purpose English model, good for creative text",
        "best_for": "Story writing, general predictions"
    },
    "BERT Base (English)": {
        "vocab_size": 30522,
        "languages": ["English"],
        "description": "Mask prediction model, understands context bidirectionally",
        "best_for": "Fill-in-the-blank, context understanding"
    },
    "BERT Multilingual": {
        "vocab_size": 119547,
        "languages": ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Hindi", "104 total"],
        "description": "Supports 104 languages",
        "best_for": "Multilingual text, language comparison"
    },
    "GPT-2 Spanish": {
        "vocab_size": 50257,
        "languages": ["Spanish"],
        "description": "Spanish language generation model",
        "best_for": "Spanish text generation"
    },
    "DistilGPT-2 (Fast)": {
        "vocab_size": 50257,
        "languages": ["English"],
        "description": "Smaller, faster version of GPT-2",
        "best_for": "Quick demos, slower devices"
    }
}

# Classroom activities
ACTIVITIES = {
    "Predict the Next Word Game": {
        "grade_level": "3-8",
        "duration": "15-20 minutes",
        "description": "Students guess the next word, then compare with AI predictions.",
        "steps": [
            "Display a sentence with the last word hidden",
            "Have students write their predictions",
            "Reveal AI's top predictions with probabilities",
            "Discuss why certain words are more likely",
            "Try changing temperature to see different predictions"
        ],
        "learning_goals": [
            "Understanding probability and prediction",
            "Pattern recognition in language",
            "Introduction to AI decision-making"
        ]
    },
    "Temperature Experiment": {
        "grade_level": "6-12",
        "duration": "25-30 minutes",
        "description": "Explore how temperature affects creativity.",
        "steps": [
            "Start with the same prompt for all students",
            "Generate predictions at temperature 0.2",
            "Generate predictions at temperature 1.5",
            "Compare and discuss differences",
            "Chart variety vs. coherence"
        ],
        "learning_goals": [
            "Understanding parameters in AI systems",
            "Balancing creativity and accuracy",
            "Data analysis and comparison"
        ]
    }
}

# Quiz questions
QUIZ_QUESTIONS = [
    {
        "question": "What does 'temperature' control in a language model?",
        "options": ["How fast the model runs", "How creative or predictable the output is", "The physical temperature of the computer", "The size of the vocabulary"],
        "correct": 1,
        "explanation": "Temperature controls randomness: low temperature is predictable, high temperature is more diverse."
    },
    {
        "question": "If a token has a probability of 0.8, what does that mean?",
        "options": ["It will definitely be chosen", "There's an 80% chance it comes next", "It's 80% correct", "The model is 80% trained"],
        "correct": 1,
        "explanation": "0.8 probability means the model assigns an 80% likelihood that token comes next."
    }
]

# === Helpers for export (PNG + PDF) ===

def _build_top_tokens_table_data(predictions: dict, top_n: int = 10):
    rows = [["Rank", "Token", "Probability"]]
    for i, (tok, p) in enumerate(list(predictions.items())[:top_n], start=1):
        rows.append([i, tok, f"{p*100:.2f}%"])
    return rows

def _draw_wrapped_text(c, text, x, y, max_width, line_height=14, font_name="Helvetica", font_size=10):
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    while words:
        w = words[0]
        test = f"{line} {w}".strip()
        if c.stringWidth(test, font_name, font_size) <= max_width:
            line = test
            words.pop(0)
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = ""
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y

def generate_pdf_report(prompt_text: str,
                        params: dict,
                        metrics: dict,
                        predictions: dict,
                        fig) -> bytes:
    chart_png = None
    if fig is not None:
        chart_png = pio.to_image(fig, format="png", width=1200, height=700, scale=2)

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    x = margin
    y = height - margin

    c.setTitle("Token Explorer Report")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Token Explorer for Educators — Prediction Report")
    y -= 18
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFillColor(colors.black)
    y -= 22
    c.line(x, y, width - margin, y)
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Input Prompt:")
    y -= 16
    y = _draw_wrapped_text(c, prompt_text or "(none)", x, y, max_width=width - 2*margin, font_size=11)

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Parameters:")
    y -= 16
    c.setFont("Helvetica", 11)
    for line in [
        f"Temperature: {params.get('temperature')}",
        f"Top-k: {params.get('top_k')}",
        f"Top-p: {params.get('top_p')}",
        f"Model: {params.get('model_name')}",
    ]:
        c.drawString(x, y, line); y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Metrics:")
    y -= 16
    c.setFont("Helvetica", 11)
    m_lines = [
        f"Entropy: {metrics.get('entropy'):.2f} bits" if metrics.get("entropy") is not None else "Entropy: n/a",
        f"Perplexity: {metrics.get('perplexity'):.2f}" if metrics.get("perplexity") is not None else "Perplexity: n/a",
        f"Top Token Probability: {max(predictions.values())*100:.1f}%" if predictions else "Top Token Probability: n/a",
    ]
    for line in m_lines:
        c.drawString(x, y, line); y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Top-10 Tokens:")
    y -= 16

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
        c.showPage()
        y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Top-10 Tokens (cont'd):")
        y -= 18

    tbl.wrapOn(c, width - 2*margin, y)
    tbl.drawOn(c, x, y - (14 * len(table_data)))
    y -= (14 * len(table_data) + 18)

    if chart_png:
        if y < margin + 220:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Probability Chart:")
        y -= 12
        img_width = width - 2*margin
        img_height = img_width * 0.55
        c.drawImage(BytesIO(chart_png), x, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
        y -= (img_height + 6)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

def export_chart_png(fig) -> bytes:
    if fig is None:
        return b""
    return pio.to_image(fig, format="png", width=1200, height=700, scale=2)

# Initialize session state
if 'tutorial_shown' not in st.session_state:
    st.session_state.tutorial_shown = False
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = False
if 'poll_mode' not in st.session_state:
    st.session_state.poll_mode = False
if 'student_predictions' not in st.session_state:
    st.session_state.student_predictions = []
if 'high_contrast' not in st.session_state:
    st.session_state.high_contrast = False
if 'font_size' not in st.session_state:
    st.session_state.font_size = 'Medium'

# Simulated tokenization function
def tokenize_text(text, model_name):
    tokens = []
    words = text.split()
    for word in words:
        if len(word) > 8 and "BERT" in model_name:
            tokens.append(word[:3]); tokens.append("##" + word[3:6])
            if len(word) > 6:
                tokens.append("##" + word[6:])
        else:
            tokens.append(word)
    return tokens

# Simulated probability generation
def generate_probabilities(prompt, model_name, temperature, top_k, top_p):
    context_predictions = {
        "The cat sat on the": {
            "mat": 0.35, "chair": 0.20, "floor": 0.15, "table": 0.12,
            "sofa": 0.08, "bed": 0.05, "couch": 0.03, "roof": 0.02
        },
        "Once upon a time in a": {
            "kingdom": 0.40, "land": 0.25, "forest": 0.15, "village": 0.10,
            "castle": 0.05, "city": 0.03, "galaxy": 0.02
        },
        "Water boils at": {
            "100": 0.70, "212": 0.15, "boiling": 0.05, "high": 0.04,
            "sea": 0.03, "room": 0.02, "atmospheric": 0.01
        },
        "To be or not to be,": {
            "that": 0.90, "this": 0.03, "whether": 0.02, "it": 0.02,
            "what": 0.01, "which": 0.01, "the": 0.01
        },
        "The Earth revolves around": {
            "the": 0.85, "Sun": 0.08, "its": 0.03, "a": 0.02,
            "our": 0.01, "itself": 0.01
        }
    }

    predictions = None
    for key in context_predictions:
        if key.lower() in prompt.lower():
            predictions = context_predictions[key].copy()
            break

    if predictions is None:
        default_tokens = ["the", "a", "and", "is", "to", "of", "in", "it", "for", "on"]
        predictions = {token: random.uniform(0.05, 0.25) for token in default_tokens}
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    if temperature > 0:
        logits = {k: math.log(v) / temperature for k, v in predictions.items()}
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        predictions = {k: v/total for k, v in exp_logits.items()}

    if top_k > 0:
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        predictions = dict(sorted_preds[:top_k])
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    if top_p < 1.0:
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        cumsum = 0
        nucleus = {}
        for token, prob in sorted_preds:
            cumsum += prob
            nucleus[token] = prob
            if cumsum >= top_p:
                break
        predictions = nucleus
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    return predictions

def calculate_entropy(probabilities):
    entropy = 0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

def calculate_perplexity(entropy):
    return 2 ** entropy

def create_probability_chart(predictions, chart_type="bar"):
    tokens = list(predictions.keys())[:10]
    probs = [predictions[t] * 100 for t in tokens]

    colors_list = []
    for p in probs:
        if p > 50: colors_list.append('#28A745')
        elif p > 20: colors_list.append('#17A2B8')
        elif p > 5: colors_list.append('#FFC107')
        else: colors_list.append('#6C757D')

    fig = go.Figure(data=[
        go.Bar(
            y=tokens,
            x=probs,
            orientation='h',
            marker=dict(color=colors_list),
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Top 10 Token Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Token",
        height=500,
        showlegend=False,
        yaxis={'categoryorder':'total ascending'}
    )
    return fig

def create_wordcloud_data(predictions):
    df = pd.DataFrame([
        {'token': token, 'probability': prob * 100}
        for token, prob in list(predictions.items())[:50]
    ])
    return df

def create_entropy_chart(entropy_values):
    fig = go.Figure(data=[
        go.Scatter(
            x=list(range(len(entropy_values))),
            y=entropy_values,
            mode='lines+markers',
            marker=dict(size=8, color='#0066CC'),
            line=dict(width=2, color='#0066CC')
        )
    ])
    fig.update_layout(
        title="Entropy Over Token Sequence",
        xaxis_title="Token Position",
        yaxis_title="Entropy (bits)",
        height=400
    )
    return fig

def main():
    st.title("🎓 Token Explorer for Educators")
    st.markdown("### Making AI Language Models Accessible to All Learners")

    if not st.session_state.tutorial_shown:
        with st.expander("👋 Welcome! Click here for a quick tour", expanded=True):
            st.markdown("""
            **Quick Start**
            1) Enter text or load an example
            2) Choose a model
            3) Adjust temperature/top-k/top-p
            4) Generate predictions
            5) Export CSV/PNG/PDF
            """)
            if st.button("Got it! Don't show this again"):
                st.session_state.tutorial_shown = True
                st.rerun()

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
    with col1:
        if st.button("📖 Glossary"):
            st.session_state.show_glossary = not st.session_state.get('show_glossary', False)
    with col2:
        if st.button("❓ Help & Tutorial"):
            st.session_state.show_tutorial = not st.session_state.show_tutorial
    with col3:
        st.session_state.poll_mode = st.checkbox("📊 Class Poll Mode", value=st.session_state.poll_mode)
    with col4:
        st.session_state.high_contrast = st.checkbox("🌓 High Contrast", value=st.session_state.high_contrast)
    with col5:
        st.session_state.font_size = st.selectbox("Font", ["Small", "Medium", "Large"],
                                                  index=["Small", "Medium", "Large"].index(st.session_state.font_size),
                                                  label_visibility="collapsed")
    st.markdown("---")

    if st.session_state.get('show_glossary', False):
        with st.expander("📖 Glossary of Terms", expanded=True):
            for term, definitions in GLOSSARY.items():
                st.markdown(f"**{term}**")
                st.markdown(f"*Simple:* {definitions['simple']}")
                st.markdown(f"*Detailed:* {definitions['detailed']}")
                st.markdown("")

    col_left, col_middle, col_right = st.columns([1, 2, 1])

    with col_left:
        st.markdown("### 📝 Input Text")
        category = st.selectbox("Load Example Prompt", ["-- Select Category --"] + list(EXAMPLE_PROMPTS.keys()))
        if category != "-- Select Category --":
            example = st.selectbox("Choose Example", EXAMPLE_PROMPTS[category])
            if st.button("Load This Example"):
                st.session_state.input_text = example
        if st.button("🎲 Random Example"):
            random_category = random.choice(list(EXAMPLE_PROMPTS.keys()))
            st.session_state.input_text = random.choice(EXAMPLE_PROMPTS[random_category])
        input_text = st.text_area("Enter your text:",
                                  value=st.session_state.get('input_text', ''),
                                  height=150)
        st.markdown("### 🤖 Model Selection")
        model_name = st.selectbox("Choose AI Model", list(MODELS.keys()))
        model_info = MODELS[model_name]
        st.info(f"**{model_name}**\n\n📊 Vocabulary: {model_info['vocab_size']:,}\n\n🌍 Languages: {', '.join(model_info['languages'][:3])}{'...' if len(model_info['languages'])>3 else ''}\n\n✨ Best for: {model_info['best_for']}")
        compare_models = st.checkbox("🔄 Compare with another model")
        model_name_2 = None
        if compare_models:
            model_name_2 = st.selectbox("Second Model", [m for m in MODELS.keys() if m != model_name])

    with col_middle:
        st.markdown("### 🎚️ Parameters")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("🛡️ Conservative"):
                st.session_state.temperature = 0.3; st.session_state.top_k = 10; st.session_state.top_p = 0.8
        with preset_col2:
            if st.button("⚖️ Balanced"):
                st.session_state.temperature = 0.8; st.session_state.top_k = 50; st.session_state.top_p = 0.9
        with preset_col3:
            if st.button("🎨 Creative"):
                st.session_state.temperature = 1.5; st.session_state.top_k = 100; st.session_state.top_p = 0.95

        temperature = st.slider("🌡️ Temperature", 0.0, 2.0, st.session_state.get('temperature', 1.0), 0.1)
        top_k = st.slider("🔝 Top-k", 0, 100, st.session_state.get('top_k', 50), 5)
        top_p = st.slider("🎯 Top-p", 0.0, 1.0, st.session_state.get('top_p', 0.9), 0.05)

        if temperature == 0:
            strategy = "🔒 Greedy Decoding"
        elif top_k > 0 and top_p < 1.0:
            strategy = f"🎯 Top-k ({top_k}) + Top-p ({top_p})"
        elif top_k > 0:
            strategy = f"🔝 Top-k ({top_k})"
        elif top_p < 1.0:
            strategy = f"🎯 Nucleus (Top-p={top_p})"
        else:
            strategy = "🌡️ Temperature Sampling"
        st.info(f"**Decoding Strategy:** {strategy}")

        if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
            if input_text.strip():
                predictions = generate_probabilities(input_text, model_name, temperature, top_k, top_p)
                st.session_state.predictions = predictions
                st.session_state.current_model = model_name
                st.session_state.current_text = input_text
                if compare_models and model_name_2:
                    predictions_2 = generate_probabilities(input_text, model_name_2, temperature, top_k, top_p)
                    st.session_state.predictions_2 = predictions_2
                    st.session_state.current_model_2 = model_name_2
                entropy = calculate_entropy(predictions)
                perplexity = calculate_perplexity(entropy)
                st.session_state.entropy = entropy
                st.session_state.perplexity = perplexity
            else:
                st.warning("Please enter some text first.")

        if 'predictions' in st.session_state:
            st.markdown("---")
            st.markdown("### 🎯 Predictions")
            predictions = st.session_state.predictions
            max_prob = max(predictions.values())
            if max_prob > 0.5:
                st.success("✅ High confidence (>50%)")
            elif max_prob > 0.2:
                st.info("ℹ️ Medium confidence (20–50%)")
            else:
                st.warning("⚠️ Low confidence (<20%)")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("📊 Entropy", f"{st.session_state.entropy:.2f} bits")
            with metric_col2:
                st.metric("🎲 Perplexity", f"{st.session_state.perplexity:.1f}")

            st.markdown("#### Top 10 Tokens")
            if compare_models and 'predictions_2' in st.session_state:
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown(f"**{st.session_state.current_model}**")
                    for i, (token, prob) in enumerate(list(predictions.items())[:10], 1):
                        pct = prob*100
                        cls = "probability-high" if pct>50 else "probability-medium" if pct>20 else "probability-low" if pct>5 else "probability-verylow"
                        st.markdown(f'<div class="{cls}">#{i}: <strong>{token}</strong> — {pct:.1f}%</div>', unsafe_allow_html=True)
                with comp_col2:
                    st.markdown(f"**{st.session_state.current_model_2}**")
                    for i, (token, prob) in enumerate(list(st.session_state.predictions_2.items())[:10], 1):
                        pct = prob*100
                        cls = "probability-high" if pct>50 else "probability-medium" if pct>20 else "probability-low" if pct>5 else "probability-verylow"
                        st.markdown(f'<div class="{cls}">#{i}: <strong>{token}</strong> — {pct:.1f}%</div>', unsafe_allow_html=True)
            else:
                for i, (token, prob) in enumerate(list(predictions.items())[:10], 1):
                    pct = prob*100
                    cls = "probability-high" if pct>50 else "probability-medium" if pct>20 else "probability-low" if pct>5 else "probability-verylow"
                    st.markdown(f'<div class="{cls}">#{i}: <strong>{token}</strong> — {pct:.1f}%</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Visualizations")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Probability Chart", "☁️ Word Cloud Data", "📈 Metrics Analysis"])

            with viz_tab1:
                fig = create_probability_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)

                # Real PNG export using plotly.io.to_image + st.download_button
                try:
                    png_bytes_inline = export_chart_png(fig)
                    st.download_button(
                        "🖼️ Download This Chart (PNG)",
                        data=png_bytes_inline,
                        file_name=f"token_probabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Chart export failed: {e}")

            with viz_tab2:
                df = create_wordcloud_data(predictions)
                st.dataframe(df, use_container_width=True)
                fig2 = px.bar(df.head(20), x='probability', y='token', orientation='h', title="Token Probability Distribution (Top 20)")
                st.plotly_chart(fig2, use_container_width=True)

            with viz_tab3:
                st.markdown(f"""
                **Model Confidence Metrics**
                - Entropy: {st.session_state.entropy:.2f} bits
                - Perplexity: {st.session_state.perplexity:.1f}
                - Top Token Probability: {max(predictions.values())*100:.1f}%
                """)

    with col_right:
        st.markdown("### 🏫 Classroom Activities")
        activity_name = st.selectbox("Choose Activity", ["-- Select Activity --"] + list(ACTIVITIES.keys()))
        if activity_name != "-- Select Activity --":
            activity = ACTIVITIES[activity_name]
            with st.expander(f"📋 {activity_name}", expanded=True):
                st.markdown(f"**Grade Level**: {activity['grade_level']}  \n**Duration**: {activity['duration']}")
                st.markdown(f"**Description**: {activity['description']}")
                st.markdown("**Steps:**")
                for i, step in enumerate(activity['steps'], 1):
                    st.markdown(f"{i}. {step}")
                st.markdown("**Learning Goals:**")
                for goal in activity['learning_goals']:
                    st.markdown(f"- {goal}")

        st.markdown("---")
        st.markdown("### 📤 Export Options")

        current_fig = None
        if 'predictions' in st.session_state:
            current_fig = create_probability_chart(st.session_state.predictions)

        # Real PNG export button for the current chart
        if 'predictions' in st.session_state and current_fig is not None:
            try:
                png_bytes = export_chart_png(current_fig)
                st.download_button(
                    label="🖼️ Download Chart Image (PNG)",
                    data=png_bytes,
                    file_name=f"token_probabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Image export failed: {e}")

        # CSV export
        if 'predictions' in st.session_state:
            df = pd.DataFrame([
                {
                    'Rank': i,
                    'Token': token,
                    'Probability': f"{prob*100:.2f}%",
                    'Model': st.session_state.get('current_model'),
                    'Temperature': st.session_state.get('temperature'),
                    'Top_k': st.session_state.get('top_k'),
                    'Top_p': st.session_state.get('top_p')
                }
                for i, (token, prob) in enumerate(st.session_state.predictions.items(), 1)
            ])
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download Predictions (CSV)",
                csv,
                file_name=f"token_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # PDF export
        if 'predictions' in st.session_state:
            try:
                pdf_bytes = generate_pdf_report(
                    prompt_text=st.session_state.get('current_text', ''),
                    params={
                        'temperature': st.session_state.get('temperature'),
                        'top_k': st.session_state.get('top_k'),
                        'top_p': st.session_state.get('top_p'),
                        'model_name': st.session_state.get('current_model')
                    },
                    metrics={
                        'entropy': st.session_state.get('entropy'),
                        'perplexity': st.session_state.get('perplexity')
                    },
                    predictions=st.session_state.predictions,
                    fig=current_fig
                )
                st.download_button(
                    label="📄 Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"token_explorer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF export failed: {e}")
        else:
            st.info("Generate predictions to enable exports.")

        st.markdown("---")
        st.markdown("### 🎓 Standards Alignment")
        with st.expander("📚 View Standards"):
            st.markdown("""
            **CSTA K-12 CS Standards**
            - 1B-AP-15, 3A-IC-24, 3B-AP-08

            **ISTE Standards for Students**
            - 1.6.d, 1.1.c

            **Common Core Math**
            - HSS-IC.A.2, 7.SP.C.7
            """)

    if st.session_state.poll_mode:
        st.markdown("---")
        st.markdown("## 📊 Class Poll Mode")
        poll_col1, poll_col2 = st.columns(2)
        with poll_col1:
            st.markdown("### 👥 Student Submissions")
            st.markdown(f"**Join Code**: POLL-{random.randint(1000, 9999)}")
            student_guess = st.text_input("Your prediction:", key="student_guess")
            if st.button("Submit Prediction"):
                if student_guess:
                    st.session_state.student_predictions.append(student_guess.lower())
                    st.success("Prediction submitted!")
            st.metric("Total Submissions", len(st.session_state.student_predictions))
            if st.button("🗑️ Clear All Predictions"):
                st.session_state.student_predictions = []
                st.rerun()
        with poll_col2:
            st.markdown("### 📊 Results Comparison")
            if st.session_state.student_predictions and 'predictions' in st.session_state:
                student_counts = Counter(st.session_state.student_predictions)
                total_students = len(st.session_state.student_predictions)
                st.markdown("**Top Student Predictions:**")
                for word, count in student_counts.most_common(5):
                    pct = (count / total_students) * 100
                    st.markdown(f"- **{word}**: {count} votes ({pct:.0f}%)")
                st.markdown("**Top AI Predictions:**")
                for token, prob in list(st.session_state.predictions.items())[:5]:
                    st.markdown(f"- **{token}**: {prob*100:.1f}%")
                top_student = student_counts.most_common(1)[0][0] if student_counts else None
                top_ai = list(st.session_state.predictions.keys())[0]
                if top_student == top_ai:
                    st.success(f"✅ Agreement on '{top_student}'")
                else:
                    st.info(f"🤔 Students: '{top_student}' vs AI: '{top_ai}'")

    if st.session_state.get('show_quiz', False):
        st.markdown("---")
        st.markdown("## 📝 Quick Knowledge Check")
        for i, q in enumerate(QUIZ_QUESTIONS):
            st.markdown(f"**Question {i+1}: {q['question']}**")
            answer = st.radio("Select your answer:", q['options'], key=f"quiz_{i}")
            if st.button(f"Check Answer #{i+1}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                else:
                    st.error(f"❌ Not quite. {q['explanation']}")
        if st.button("Close Quiz"):
            st.session_state.show_quiz = False
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6C757D; padding: 20px;'>
      <p><strong>Token Explorer for Educators</strong> | Version 2.0 | November 2025</p>
      <p>Making AI Accessible to All Learners</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
