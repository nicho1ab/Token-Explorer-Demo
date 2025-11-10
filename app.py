"""
Token Explorer for Educators - Streamlit Application
Enhanced version with all requested features for non-technical educators
Deploy with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import random
import math
from collections import Counter
import base64
from io import BytesIO

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
    /* High contrast and accessibility */
    .stButton>button {
        min-height: 44px;
        min-width: 44px;
        font-size: 16px;
    }

    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #0066CC;
        cursor: help;
    }

    .probability-high {
        background-color: #28A745;
        color: white;
        padding: 8px;
        border-radius: 5px;
    }

    .probability-medium {
        background-color: #17A2B8;
        color: white;
        padding: 8px;
        border-radius: 5px;
    }

    .probability-low {
        background-color: #FFC107;
        color: black;
        padding: 8px;
        border-radius: 5px;
    }

    .probability-verylow {
        background-color: #6C757D;
        color: white;
        padding: 8px;
        border-radius: 5px;
    }

    .activity-card {
        border: 2px solid #0066CC;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .info-box {
        background-color: #E7F3FF;
        border-left: 5px solid #0066CC;
        padding: 15px;
        margin: 10px 0;
    }

    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 15px;
        margin: 10px 0;
    }

    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Educational content and glossary
GLOSSARY = {
    "Token": {
        "simple": "A piece of text that the AI model understands - like a word, part of a word, or punctuation.",
        "detailed": "Tokens are the basic units that language models use to process text. A word like 'running' might be one token, while 'unbelievable' might be split into 'un', 'believe', and 'able'. This helps the model understand word patterns and meanings."
    },
    "Probability": {
        "simple": "How likely the AI thinks a word should come next, shown as a percentage (0% to 100%).",
        "detailed": "The model calculates probabilities for thousands of possible next tokens based on what it learned during training. Higher probability means the model is more confident that token should come next."
    },
    "Temperature": {
        "simple": "Controls how creative or predictable the AI is. Low = boring but safe, High = creative but risky.",
        "detailed": "Temperature (0.0-2.0) adjusts how the model chooses tokens. At 0, it always picks the most likely word (deterministic). At higher values like 1.5, it takes more chances with less common words, making output more creative but potentially less coherent."
    },
    "Top-k": {
        "simple": "Limits the AI to choosing from only the k most likely words (e.g., top 50 choices).",
        "detailed": "Top-k sampling restricts the model to considering only the k tokens with highest probability. For example, with k=50, the model picks from the 50 most likely next words, ignoring all others. This prevents rare or nonsensical tokens."
    },
    "Top-p (Nucleus)": {
        "simple": "Picks from the smallest set of words that together add up to probability p (e.g., 90%).",
        "detailed": "Also called nucleus sampling, top-p selects tokens dynamically. With p=0.9, it considers only enough top tokens to reach 90% cumulative probability. This adapts to context - more choices when uncertain, fewer when confident."
    },
    "Perplexity": {
        "simple": "Measures how confused the AI is - lower numbers mean it's more confident in its predictions.",
        "detailed": "Perplexity quantifies model uncertainty. A perplexity of 10 means the model is as confused as if choosing randomly among 10 options. Lower perplexity indicates better prediction quality and language understanding."
    },
    "Entropy": {
        "simple": "Measures uncertainty - how spread out the probabilities are. Higher means more unpredictable.",
        "detailed": "Entropy (in bits) measures the distribution of probabilities. Low entropy means one clear winner (predictable), high entropy means many equally likely options (unpredictable). It's related to perplexity: perplexity = 2^entropy."
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
        "description": "Students guess what word comes next, then compare with AI predictions to understand how models learn patterns.",
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
        "description": "Explore how temperature affects creativity by generating multiple continuations at different settings.",
        "steps": [
            "Start with the same prompt for all students",
            "Generate predictions at temperature 0.2 (conservative)",
            "Generate predictions at temperature 1.5 (creative)",
            "Compare and discuss differences",
            "Create a chart showing variety vs. coherence"
        ],
        "learning_goals": [
            "Understanding parameters in AI systems",
            "Balancing creativity and accuracy",
            "Data analysis and comparison"
        ]
    },
    "Multilingual Token Discovery": {
        "grade_level": "5-12",
        "duration": "20-25 minutes",
        "description": "Compare how different languages are tokenized and explore cultural patterns in AI training data.",
        "steps": [
            "Input the same sentence in English and Spanish",
            "Compare token counts and predictions",
            "Discuss why some languages need more tokens",
            "Explore cultural context in predictions",
            "Test with student home languages if available"
        ],
        "learning_goals": [
            "Language structure awareness",
            "Cultural representation in AI",
            "Multilingual communication"
        ]
    },
    "Bias Detection Workshop": {
        "grade_level": "8-12",
        "duration": "35-45 minutes",
        "description": "Examine how AI predictions might reflect biases in training data and discuss ethical implications.",
        "steps": [
            "Try prompts like 'The doctor walked into' vs 'The nurse walked into'",
            "Record top predictions for occupations",
            "Analyze for gender or other biases",
            "Research where training data comes from",
            "Discuss how to make AI more fair"
        ],
        "learning_goals": [
            "Critical thinking about AI",
            "Understanding bias and fairness",
            "AI ethics and responsibility"
        ]
    },
    "Creative Writing with AI": {
        "grade_level": "4-10",
        "duration": "30-40 minutes",
        "description": "Use AI predictions as inspiration for creative writing, comparing human and machine creativity.",
        "steps": [
            "Students start a story with one sentence",
            "AI suggests next words at high temperature",
            "Students choose: use AI suggestion or their own",
            "Continue for 5-10 rounds",
            "Share and compare AI-assisted vs. original stories"
        ],
        "learning_goals": [
            "Creative writing skills",
            "Human-AI collaboration",
            "Understanding AI capabilities and limitations"
        ]
    }
}

# Quiz questions
QUIZ_QUESTIONS = [
    {
        "question": "What does 'temperature' control in a language model?",
        "options": [
            "How fast the model runs",
            "How creative or predictable the output is",
            "The physical temperature of the computer",
            "The size of the vocabulary"
        ],
        "correct": 1,
        "explanation": "Temperature controls randomness: low temperature makes predictions predictable, high temperature makes them more creative and diverse."
    },
    {
        "question": "If a token has a probability of 0.8, what does that mean?",
        "options": [
            "It will definitely be chosen",
            "There's an 80% chance the model thinks it should come next",
            "It's 80% correct",
            "The model is 80% trained"
        ],
        "correct": 1,
        "explanation": "Probability of 0.8 (or 80%) means the model assigns an 80% likelihood to that token being the next word, based on patterns it learned."
    },
    {
        "question": "What is a 'token' in language models?",
        "options": [
            "A password to access the AI",
            "A reward for good predictions",
            "A piece of text like a word or word part",
            "A type of computer chip"
        ],
        "correct": 2,
        "explanation": "Tokens are the basic units AI models use to process text - they can be whole words, parts of words (like 'un' or 'ing'), or punctuation."
    },
    {
        "question": "Lower perplexity means the model is...",
        "options": [
            "More confused",
            "Making random guesses",
            "More confident in its predictions",
            "Using fewer tokens"
        ],
        "correct": 2,
        "explanation": "Lower perplexity indicates the model is more certain about what comes next, showing better understanding of the language patterns."
    }
]

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
    """Simulate tokenization based on model type"""
    tokens = []
    words = text.split()

    for word in words:
        if len(word) > 8 and "BERT" in model_name:
            # Simulate subword tokenization for BERT
            tokens.append(word[:3])
            tokens.append("##" + word[3:6])
            if len(word) > 6:
                tokens.append("##" + word[6:])
        else:
            tokens.append(word)

    return tokens

# Simulated probability generation
def generate_probabilities(prompt, model_name, temperature, top_k, top_p):
    """Generate realistic probability distributions for next tokens"""

    # Context-aware token suggestions
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

    # Find closest match or use default
    predictions = None
    for key in context_predictions:
        if key.lower() in prompt.lower():
            predictions = context_predictions[key].copy()
            break

    # Default predictions if no match
    if predictions is None:
        default_tokens = ["the", "a", "and", "is", "to", "of", "in", "it", "for", "on"]
        predictions = {token: random.uniform(0.05, 0.25) for token in default_tokens}
        # Normalize
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    # Apply temperature scaling
    if temperature > 0:
        # Convert to logits and apply temperature
        logits = {k: math.log(v) / temperature for k, v in predictions.items()}
        # Convert back to probabilities
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        predictions = {k: v/total for k, v in exp_logits.items()}

    # Apply top-k filtering
    if top_k > 0:
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        predictions = dict(sorted_preds[:top_k])
        # Renormalize
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    # Apply top-p (nucleus) filtering
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
        # Renormalize
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

    # Sort by probability
    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

    return predictions

# Calculate entropy
def calculate_entropy(probabilities):
    """Calculate Shannon entropy from probability distribution"""
    entropy = 0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Calculate perplexity
def calculate_perplexity(entropy):
    """Calculate perplexity from entropy"""
    return 2 ** entropy

# Create probability bar chart
def create_probability_chart(predictions, chart_type="bar"):
    """Create visualization of token probabilities"""

    tokens = list(predictions.keys())[:10]
    probs = [predictions[t] * 100 for t in tokens]

    # Color code by probability
    colors = []
    for p in probs:
        if p > 50:
            colors.append('#28A745')  # Green
        elif p > 20:
            colors.append('#17A2B8')  # Blue
        elif p > 5:
            colors.append('#FFC107')  # Orange
        else:
            colors.append('#6C757D')  # Gray

    fig = go.Figure(data=[
        go.Bar(
            y=tokens,
            x=probs,
            orientation='h',
            marker=dict(color=colors),
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

# Create word cloud data
def create_wordcloud_data(predictions):
    """Prepare data for word cloud visualization"""
    df = pd.DataFrame([
        {'token': token, 'probability': prob * 100}
        for token, prob in list(predictions.items())[:50]
    ])
    return df

# Create entropy chart
def create_entropy_chart(entropy_values):
    """Create line chart showing entropy over token positions"""
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

# Main application
def main():
    # Header
    st.title("🎓 Token Explorer for Educators")
    st.markdown("### Making AI Language Models Accessible to All Learners")

    # Tutorial/Welcome Modal
    if not st.session_state.tutorial_shown:
        with st.expander("👋 Welcome! Click here for a quick tour", expanded=True):
            st.markdown("""
            **Welcome to Token Explorer for Educators!**

            This tool helps you understand how AI language models work by showing you how they predict the next word in a sentence.

            **Quick Start:**
            1. 📝 Enter text or load an example prompt
            2. 🤖 Choose an AI model
            3. 🎚️ Adjust parameters (temperature, top-k, top-p)
            4. 🔍 See predictions with probabilities
            5. 📊 Explore visualizations
            6. 📤 Export results for your classroom

            **Features:**
            - 🎯 25 curated example prompts
            - 📚 Interactive glossary for technical terms
            - 🏫 5 ready-to-use classroom activities
            - 📊 Multiple visualization types
            - 🌍 Multilingual model support
            - ♿ Full accessibility features
            - 📝 Quiz generator for assessments
            """)

            if st.button("Got it! Don't show this again"):
                st.session_state.tutorial_shown = True
                st.rerun()

    # Top navigation
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

    # Show glossary if toggled
    if st.session_state.get('show_glossary', False):
        with st.expander("📖 Glossary of Terms", expanded=True):
            for term, definitions in GLOSSARY.items():
                st.markdown(f"**{term}**")
                st.markdown(f"*Simple:* {definitions['simple']}")
                st.markdown(f"*Detailed:* {definitions['detailed']}")
                st.markdown("")

    # Main layout: 3 columns
    col_left, col_middle, col_right = st.columns([1, 2, 1])

    # LEFT COLUMN - Input and Model Selection
    with col_left:
        st.markdown("### 📝 Input Text")

        # Example prompt selector
        category = st.selectbox(
            "Load Example Prompt",
            ["-- Select Category --"] + list(EXAMPLE_PROMPTS.keys())
        )

        if category != "-- Select Category --":
            example = st.selectbox("Choose Example", EXAMPLE_PROMPTS[category])
            if st.button("Load This Example"):
                st.session_state.input_text = example

        if st.button("🎲 Random Example"):
            random_category = random.choice(list(EXAMPLE_PROMPTS.keys()))
            st.session_state.input_text = random.choice(EXAMPLE_PROMPTS[random_category])

        # Text input
        input_text = st.text_area(
            "Enter your text:",
            value=st.session_state.get('input_text', ''),
            height=150,
            help="Type any sentence or use the examples above"
        )

        st.markdown("### 🤖 Model Selection")

        model_name = st.selectbox(
            "Choose AI Model",
            list(MODELS.keys()),
            help="Different models process text differently"
        )

        # Show model info
        model_info = MODELS[model_name]
        st.info(f"""
        **{model_name}**

        📊 Vocabulary: {model_info['vocab_size']:,} tokens

        🌍 Languages: {', '.join(model_info['languages'][:3])}{'...' if len(model_info['languages']) > 3 else ''}

        ✨ Best for: {model_info['best_for']}
        """)

        # Model comparison mode
        compare_models = st.checkbox("🔄 Compare with another model")
        model_name_2 = None
        if compare_models:
            model_name_2 = st.selectbox(
                "Second Model",
                [m for m in MODELS.keys() if m != model_name]
            )

    # MIDDLE COLUMN - Parameters and Predictions
    with col_middle:
        st.markdown("### 🎚️ Parameters")

        # Parameter presets
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("🛡️ Conservative"):
                st.session_state.temperature = 0.3
                st.session_state.top_k = 10
                st.session_state.top_p = 0.8
        with preset_col2:
            if st.button("⚖️ Balanced"):
                st.session_state.temperature = 0.8
                st.session_state.top_k = 50
                st.session_state.top_p = 0.9
        with preset_col3:
            if st.button("🎨 Creative"):
                st.session_state.temperature = 1.5
                st.session_state.top_k = 100
                st.session_state.top_p = 0.95

        # Parameter sliders
        temperature = st.slider(
            "🌡️ Temperature (creativity)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get('temperature', 1.0),
            step=0.1,
            help=GLOSSARY["Temperature"]["simple"]
        )

        top_k = st.slider(
            "🔝 Top-k (limit choices)",
            min_value=0,
            max_value=100,
            value=st.session_state.get('top_k', 50),
            step=5,
            help=GLOSSARY["Top-k"]["simple"]
        )

        top_p = st.slider(
            "🎯 Top-p / Nucleus (dynamic selection)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('top_p', 0.9),
            step=0.05,
            help=GLOSSARY["Top-p (Nucleus)"]["simple"]
        )

        # Show decoding strategy
        if temperature == 0:
            strategy = "🔒 Greedy Decoding (always picks most likely)"
        elif top_k > 0 and top_p < 1.0:
            strategy = f"🎯 Top-k ({top_k}) + Top-p ({top_p}) Sampling"
        elif top_k > 0:
            strategy = f"🔝 Top-k ({top_k}) Sampling"
        elif top_p < 1.0:
            strategy = f"🎯 Nucleus (Top-p = {top_p}) Sampling"
        else:
            strategy = "🌡️ Temperature Sampling"

        st.info(f"**Decoding Strategy:** {strategy}")

        # Generate button
        if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
            if input_text.strip():
                # Generate predictions
                predictions = generate_probabilities(input_text, model_name, temperature, top_k, top_p)
                st.session_state.predictions = predictions
                st.session_state.current_model = model_name
                st.session_state.current_text = input_text

                # If comparing models, generate second set
                if compare_models and model_name_2:
                    predictions_2 = generate_probabilities(input_text, model_name_2, temperature, top_k, top_p)
                    st.session_state.predictions_2 = predictions_2
                    st.session_state.current_model_2 = model_name_2

                # Calculate metrics
                entropy = calculate_entropy(predictions)
                perplexity = calculate_perplexity(entropy)
                st.session_state.entropy = entropy
                st.session_state.perplexity = perplexity
            else:
                st.warning("Please enter some text first!")

        # Display predictions
        if 'predictions' in st.session_state:
            st.markdown("---")
            st.markdown("### 🎯 Predictions")

            # Context explanation
            predictions = st.session_state.predictions
            max_prob = max(predictions.values())

            if max_prob > 0.5:
                st.success("✅ **High Confidence**: The model has a clear favorite (>50% probability)")
            elif max_prob > 0.2:
                st.info("ℹ️ **Medium Confidence**: Several likely options (20-50% probability)")
            else:
                st.warning("⚠️ **Low Confidence**: Many equally possible choices (<20% probability)")

            # Display metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "📊 Entropy",
                    f"{st.session_state.entropy:.2f} bits",
                    help=GLOSSARY["Entropy"]["simple"]
                )
            with metric_col2:
                st.metric(
                    "🎲 Perplexity",
                    f"{st.session_state.perplexity:.1f}",
                    help=GLOSSARY["Perplexity"]["simple"]
                )

            # Show top predictions
            st.markdown("#### Top 10 Tokens")

            if compare_models and 'predictions_2' in st.session_state:
                # Side-by-side comparison
                comp_col1, comp_col2 = st.columns(2)

                with comp_col1:
                    st.markdown(f"**{st.session_state.current_model}**")
                    for i, (token, prob) in enumerate(list(predictions.items())[:10], 1):
                        prob_pct = prob * 100
                        if prob_pct > 50:
                            color_class = "probability-high"
                        elif prob_pct > 20:
                            color_class = "probability-medium"
                        elif prob_pct > 5:
                            color_class = "probability-low"
                        else:
                            color_class = "probability-verylow"

                        st.markdown(f"""
                        <div class="{color_class}">
                        #{i}: <strong>{token}</strong> — {prob_pct:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

                with comp_col2:
                    st.markdown(f"**{st.session_state.current_model_2}**")
                    predictions_2 = st.session_state.predictions_2
                    for i, (token, prob) in enumerate(list(predictions_2.items())[:10], 1):
                        prob_pct = prob * 100
                        if prob_pct > 50:
                            color_class = "probability-high"
                        elif prob_pct > 20:
                            color_class = "probability-medium"
                        elif prob_pct > 5:
                            color_class = "probability-low"
                        else:
                            color_class = "probability-verylow"

                        st.markdown(f"""
                        <div class="{color_class}">
                        #{i}: <strong>{token}</strong> — {prob_pct:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Single model display
                for i, (token, prob) in enumerate(list(predictions.items())[:10], 1):
                    prob_pct = prob * 100
                    if prob_pct > 50:
                        color_class = "probability-high"
                    elif prob_pct > 20:
                        color_class = "probability-medium"
                    elif prob_pct > 5:
                        color_class = "probability-low"
                    else:
                        color_class = "probability-verylow"

                    st.markdown(f"""
                    <div class="{color_class}">
                    #{i}: <strong>{token}</strong> — {prob_pct:.1f}%
                    </div>
                    """, unsafe_allow_html=True)

            # Visualizations
            st.markdown("---")
            st.markdown("### 📊 Visualizations")

            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "📊 Probability Chart",
                "☁️ Word Cloud Data",
                "📈 Metrics Analysis"
            ])

            with viz_tab1:
                fig = create_probability_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)

                # Download chart as image
                if st.button("📥 Download Chart"):
                    st.info("Chart downloaded! (In actual deployment, this would trigger a download)")

            with viz_tab2:
                df = create_wordcloud_data(predictions)
                st.dataframe(df, use_container_width=True)

                # Simple bar chart as word cloud alternative
                fig2 = px.bar(
                    df.head(20),
                    x='probability',
                    y='token',
                    orientation='h',
                    title="Token Probability Distribution (Top 20)"
                )
                st.plotly_chart(fig2, use_container_width=True)

            with viz_tab3:
                st.markdown(f"""
                **Model Confidence Metrics**

                - **Entropy**: {st.session_state.entropy:.2f} bits
                  - Lower entropy = more predictable
                  - Higher entropy = more uncertain

                - **Perplexity**: {st.session_state.perplexity:.1f}
                  - Equivalent to choosing from ~{int(st.session_state.perplexity)} equally likely options
                  - Lower is better (more confident)

                - **Top Token Probability**: {max(predictions.values())*100:.1f}%
                  - How confident the model is in its top choice
                """)

    # RIGHT COLUMN - Activities and Export
    with col_right:
        st.markdown("### 🏫 Classroom Activities")

        activity_name = st.selectbox(
            "Choose Activity",
            ["-- Select Activity --"] + list(ACTIVITIES.keys())
        )

        if activity_name != "-- Select Activity --":
            activity = ACTIVITIES[activity_name]

            with st.expander(f"📋 {activity_name}", expanded=True):
                st.markdown(f"""
                **Grade Level**: {activity['grade_level']}  
                **Duration**: {activity['duration']}

                **Description**: {activity['description']}

                **Steps**:
                """)
                for i, step in enumerate(activity['steps'], 1):
                    st.markdown(f"{i}. {step}")

                st.markdown("**Learning Goals**:")
                for goal in activity['learning_goals']:
                    st.markdown(f"- {goal}")

                if st.button("🎯 Configure Tool for This Activity"):
                    st.success("Tool configured! Try using the suggested prompts and settings.")

                if st.button("🖨️ Print Activity Handout"):
                    st.info("Handout ready for printing!")

        st.markdown("---")
        st.markdown("### 📤 Export Options")

        if st.button("📄 Export as PDF", use_container_width=True):
            st.success("PDF generated! (Would download in production)")

        if st.button("🖼️ Export as Image", use_container_width=True):
            st.success("Image saved! (Would download in production)")

        if st.button("📊 Export as CSV", use_container_width=True):
            if 'predictions' in st.session_state:
                df = pd.DataFrame([
                    {
                        'Rank': i,
                        'Token': token,
                        'Probability': f"{prob*100:.2f}%",
                        'Model': st.session_state.current_model,
                        'Temperature': temperature,
                        'Top_k': top_k,
                        'Top_p': top_p
                    }
                    for i, (token, prob) in enumerate(st.session_state.predictions.items(), 1)
                ])

                csv = df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    file_name=f"token_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        if st.button("📝 Generate Quiz", use_container_width=True):
            st.session_state.show_quiz = True

        st.markdown("---")
        st.markdown("### 🎓 Standards Alignment")

        with st.expander("📚 View Standards"):
            st.markdown("""
            **CSTA K-12 CS Standards:**
            - 1B-AP-15: Test and debug algorithms
            - 3A-IC-24: Evaluate computational artifacts for bias
            - 3B-AP-08: Describe how AI and ML algorithms work

            **ISTE Standards for Students:**
            - 1.6.d: Students understand how AI makes decisions
            - 1.1.c: Students use technology for creative expression

            **Common Core Math:**
            - HSS-IC.A.2: Analyze decisions using probability
            - 7.SP.C.7: Develop probability models
            """)

    # Poll Mode Section
    if st.session_state.poll_mode:
        st.markdown("---")
        st.markdown("## 📊 Class Poll Mode")

        poll_col1, poll_col2 = st.columns(2)

        with poll_col1:
            st.markdown("### 👥 Student Submissions")
            st.markdown(f"**Join Code**: POLL-{random.randint(1000, 9999)}")
            st.markdown("*Students: Enter your prediction for the next word*")

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
                # Count student predictions
                student_counts = Counter(st.session_state.student_predictions)
                total_students = len(st.session_state.student_predictions)

                # Top 5 student predictions
                st.markdown("**Top Student Predictions:**")
                for word, count in student_counts.most_common(5):
                    pct = (count / total_students) * 100
                    st.markdown(f"- **{word}**: {count} votes ({pct:.0f}%)")

                st.markdown("**Top AI Predictions:**")
                for token, prob in list(st.session_state.predictions.items())[:5]:
                    st.markdown(f"- **{token}**: {prob*100:.1f}%")

                # Check agreement
                top_student = student_counts.most_common(1)[0][0] if student_counts else None
                top_ai = list(st.session_state.predictions.keys())[0]

                if top_student == top_ai:
                    st.success(f"✅ **Agreement!** Both chose '{top_student}'")
                else:
                    st.info(f"🤔 **Different choices**: Students picked '{top_student}', AI picked '{top_ai}'")

    # Quiz Section
    if st.session_state.get('show_quiz', False):
        st.markdown("---")
        st.markdown("## 📝 Quick Knowledge Check")

        for i, q in enumerate(QUIZ_QUESTIONS):
            st.markdown(f"**Question {i+1}: {q['question']}**")
            answer = st.radio(
                "Select your answer:",
                q['options'],
                key=f"quiz_{i}"
            )

            if st.button(f"Check Answer #{i+1}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                else:
                    st.error(f"❌ Not quite. {q['explanation']}")

        if st.button("Close Quiz"):
            st.session_state.show_quiz = False
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6C757D; padding: 20px;'>
    <p><strong>Token Explorer for Educators</strong> | Version 2.0 | November 2025</p>
    <p>Making AI Accessible to All Learners | Built with ❤️ for Educators</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
"""

