import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer
)

# Sidebar model options
MODEL_OPTIONS = {
    "distilgpt2": ("distilgpt2", AutoModelForCausalLM, AutoTokenizer),
    "gpt2": ("gpt2", AutoModelForCausalLM, AutoTokenizer),
    "bert-base-uncased": ("bert-base-uncased", BertForMaskedLM, BertTokenizer),
}

st.set_page_config(
    page_title="Token Explorer for Educators",
    layout="wide"
)

# ----- EXPLANATION BOX -----
st.title("Token Explorer for Educators")
st.markdown(
    "#### Visualize how AI models pick the next word, one token at a time."
)
st.info("""
**How does an AI model generate text?**

Each token is chosen by sampling from a probability distribution over all possible next tokens. *Temperature* controls how random the choice is — a lower temperature means the model picks more likely tokens, while a higher temperature adds diversity and unpredictability.

This tool lets you step through generation, seeing probabilities, choices, and cumulative sentence likelihood — no coding required!
""")

# ----- SIDEBAR -----
with st.sidebar:
    st.header("Settings")
    chosen_model = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        format_func=lambda k: k.replace("-", " ").title()
    )
    top_k = st.slider("Top-k (show only top K candidates)", 1, 20, value=5)
    temperature = st.slider("Temperature", 0.1, 2.0, value=1.0, step=0.05)
    show_prob_chart = st.checkbox("Show probabilities as chart", value=True)
    show_token_ids = st.checkbox("Show token IDs (advanced)", value=False)
    show_logprobs = st.checkbox("Show log-probs (advanced)", value=False)
    guess_mode = st.checkbox("Enable 'Guess the Next Token' mode", value=False)

# ----- CACHE MODEL & TOKENIZER -----
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model_id, model_cls, tokenizer_cls = MODEL_OPTIONS[model_name]
    model = model_cls.from_pretrained(model_id)
    tokenizer = tokenizer_cls.from_pretrained(model_id)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(chosen_model)

# ----- SESSION STATE -----
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "probabilities" not in st.session_state:
    st.session_state.probabilities = []
if "cumulative_probs" not in st.session_state:
    st.session_state.cumulative_probs = []
if "last_logits" not in st.session_state:
    st.session_state.last_logits = None

# ----- MAIN AREA -----
st.markdown("### Start Your Sentence")
input_text = st.text_input(
    "Enter the beginning of your sentence:",
    value=st.session_state.input_text,
    key="main_input",
    help="Type your sentence fragment for prediction."
)
if st.button("Reset", help="Clear generated tokens and start over."):
    st.session_state.input_text = ""
    st.session_state.generated_tokens = []
    st.session_state.probabilities = []
    st.session_state.cumulative_probs = []
    st.session_state.last_logits = None
    input_text = ""

def softmax(x, temperature=1.0):
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def decode_tokens(tokens, tokenizer):
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(tokens)
    return ' '.join([tokenizer.ids_to_tokens.get(t, str(t)) for t in tokens])

def prediction_step(
    model, tokenizer, input_text, temperature=1.0, top_k=5, is_bert=False
):
    if is_bert:
        # Bert uses masked tokens for prediction
        tokens = tokenizer.encode(input_text + " [MASK]", return_tensors="pt")
        mask_index = (tokens == tokenizer.mask_token_id).nonzero()[0, 1].item()
        with torch.no_grad():
            outputs = model(tokens)
        logits = outputs.logits[0, mask_index]
    else:
        tokens = tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(tokens)
        logits = outputs.logits[0, -1]
    probs = softmax(logits.cpu().numpy(), temperature)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_tokens = [tokenizer.decode([i]) for i in top_indices]
    top_probs = probs[top_indices]
    top_ids = top_indices.tolist()
    selected_idx = np.random.choice(top_indices, p=top_probs / top_probs.sum())
    selected_token = tokenizer.decode([selected_idx])
    selected_prob = probs[selected_idx]
    return {
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "top_ids": top_ids,
        "selected_token": selected_token,
        "selected_prob": selected_prob,
        "logits": logits,
    }

is_bert = chosen_model.startswith("bert")
if st.button("Predict Next Token"):
    pred = prediction_step(
        model, tokenizer, input_text, temperature, top_k, is_bert
    )
    st.session_state.input_text = input_text
    st.session_state.last_logits = pred["logits"]
    st.session_state.generated_tokens = [pred["selected_token"]]
    st.session_state.probabilities = [pred["selected_prob"]]
    st.session_state.cumulative_probs = [pred["selected_prob"]]
    token_display = [pred["selected_token"]]
elif st.button("Continue One Token"):
    # Compose previous tokens into new input
    previous_text = st.session_state.input_text + "".join(st.session_state.generated_tokens)
    pred = prediction_step(
        model, tokenizer, previous_text, temperature, top_k, is_bert
    )
    st.session_state.generated_tokens.append(pred["selected_token"])
    st.session_state.probabilities.append(pred["selected_prob"])
    cumulative = np.prod(st.session_state.probabilities)
    st.session_state.cumulative_probs.append(cumulative)
    st.session_state.last_logits = pred["logits"]
    token_display = st.session_state.generated_tokens
else:
    token_display = st.session_state.generated_tokens

# ----- GUESS MODE -----
if guess_mode and st.session_state.last_logits is not None:
    st.markdown("#### Guess the Next Token!")
    guess = st.text_input(
        "Type your guess for the next word/token:",
        value="", key="guess_input"
    )
    logits = st.session_state.last_logits.cpu().numpy()
    probs = softmax(logits, temperature)
    # Find matching token index
    guess_token_id = None
    if guess:
        try:
            guess_token_id = tokenizer.encode(guess, add_special_tokens=False)[0]
        except Exception:
            pass
    if guess_token_id is not None:
        guess_prob = probs[guess_token_id]
        st.write(f"Your guess: '{guess}' is token ID {guess_token_id}")
        st.write(f"Model's probability for your guess: {guess_prob:.5f}")
    else:
        st.write("Invalid or unrecognized token for this model.")

# ----- PROBABILITIES CHART -----
def plot_probabilities(tokens, probs, top_ids, selected_idx, show_ids, show_logprobs):
    colors = []
    for i, p in enumerate(probs):
        if p > 0.5:
            colors.append("green")
        elif p > 0.1:
            colors.append("yellow")
        else:
            colors.append("red")
    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, probs, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    display_labels = []
    for i, tok in enumerate(tokens):
        label = f"{tok}"
        if show_ids:
            label += f" [{top_ids[i]}]"
        if show_logprobs:
            logp = np.log(probs[i] + 1e-9)
            label += f" ({logp:.2f})"
        display_labels.append(label)
    ax.set_yticklabels(display_labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Probability", fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Highlight selected token
    ax.barh(selected_idx, probs[selected_idx], color="lightblue", edgecolor="black")
    st.pyplot(fig)

if st.session_state.last_logits is not None:
    pred_logits = st.session_state.last_logits
    pred_probs = softmax(pred_logits.cpu().numpy(), temperature)
    top_indices = np.argsort(pred_probs)[::-1][:top_k]
    top_tokens = [tokenizer.decode([i]) for i in top_indices]
    top_probs = pred_probs[top_indices]
    top_ids = top_indices.tolist()
    selected_idx = np.argmax(top_probs)
    if show_prob_chart:
        st.markdown("#### Top-k Token Probabilities")
        plot_probabilities(
            top_tokens, top_probs, top_ids, selected_idx,
            show_ids=show_token_ids, show_logprobs=show_logprobs
        )
    st.markdown(
        f"**Model's top choice:** <span style='color:lightblue'>{top_tokens[selected_idx]}</span> ({top_probs[selected_idx]:.3f})",
        unsafe_allow_html=True
    )

# ----- GENERATED TOKENS TABLE -----
if token_display:
    st.markdown("#### Generated Tokens (step by step)")
    table_data = []
    for i, (tok, prob) in enumerate(zip(token_display, st.session_state.probabilities)):
        color = (
            "green" if prob > 0.5 else
            "yellow" if prob > 0.1 else
            "red"
        )
        table_data.append(
            [i + 1, f"<span style='color:{color}'>{tok}</span>", f"{prob:.3f}"]
        )
    st.markdown(
        "<table><tr><th>Step</th><th>Token</th><th>Probability</th></tr>"
        +
        "".join(
            f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
            for row in table_data
        )
        +
        "</table>",
        unsafe_allow_html=True
    )

# ----- CUMULATIVE PROBABILITY PLOT -----
if st.session_state.cumulative_probs:
    st.markdown("#### Cumulative Sentence Probability")
    fig2, ax2 = plt.subplots(figsize=(8,2))
    ax2.plot(
        np.arange(1, len(st.session_state.cumulative_probs) + 1),
        st.session_state.cumulative_probs,
        marker="o", color="purple"
    )
    ax2.set_xlabel("Generation Step", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True)
    st.pyplot(fig2)

# ----- END -----

st.markdown("---")
st.markdown(
    "<div style='font-size:16px;text-align:center;'>Built for educators. No code required.</div>",
    unsafe_allow_html=True
)
