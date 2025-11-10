# Token Explorer for Educators

Visualize how AI models pick the next word, one token at a time.

![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**Token Explorer for Educators** is an interactive web application designed to help non-technical educators understand how large language models (LLMs) generate text. Instead of treating AI as "magic," this tool demystifies the token-by-token generation process by showing:

- **How models choose words**: Real probability distributions over candidate tokens
- **Temperature effects**: How randomness vs. determinism changes output
- **Cumulative likelihood**: How overall sentence probability decreases with each token
- **Interactive learning**: Step through generation one token at a time with full visibility

Perfect for classroom demonstrations, workshops, and educational outreach about AI literacy.

## Features

✨ **Core Functionality**
- Support for multiple pre-trained models (distilgpt2, gpt2, bert-base-uncased)
- Real-time probability distribution visualization
- Step-by-step token generation with user control
- Temperature and Top-k sampling controls
- Color-coded token confidence (green = high, yellow = medium, red = low)

🎓 **Educator-Friendly Design**
- Clean, single-page interface—no code or terminal needed
- Large, legible fonts optimized for projector presentations
- Built-in explanation of how tokenization and probability work
- One-click deployment to the web

🔍 **Advanced Features** (optional toggles)
- Show token IDs for deeper analysis
- Display log-probabilities for technical audiences
- "Guess the Next Token" mode for audience engagement

## Live Demo

Try the app online: [Token Explorer on Streamlit Community Cloud](https://go.uic.edu/Token-Explorer) 

## Getting Started Locally

### Requirements

- Python 3.8 or higher
- ~2 GB RAM (models are relatively lightweight)
- CPU or GPU (GPU recommended but not required)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/token-explorer-educators.git
   cd token-explorer-educators
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

### Basic Workflow

1. **Enter a sentence fragment** in the text input (e.g., "The quick brown")
2. **Click "Predict Next Token"** to see the model's top-k predictions with probabilities
3. **Adjust Temperature** (left sidebar) to see how randomness affects choices
   - *Low temperature* (0.1): Model picks most likely tokens
   - *High temperature* (2.0): Model explores unlikely options
4. **Click "Continue One Token"** to step through generation interactively
5. **Watch the probability chart and cumulative likelihood** update in real time

### Settings (Sidebar)

| Setting | Purpose |
|---------|---------|
| **Model** | Choose from distilgpt2 (fast), gpt2 (accurate), or bert-base-uncased |
| **Top-k** | Show only the top K most likely tokens (default: 5) |
| **Temperature** | Control randomness in sampling (0.1–2.0, default: 1.0) |
| **Show probabilities as chart** | Toggle the bar chart visualization |
| **Show token IDs** | Display internal token indices (advanced) |
| **Show log-probs** | Display log-probabilities (advanced) |
| **Enable Guess Mode** | Let students/audience guess the next token |

## Deployment

### Deploy to Streamlit Community Cloud (Easiest)

1. **Push this repository to GitHub** (if not already done)

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

3. **Click "Create app"** and select:
   - **Repository**: `YOUR_USERNAME/token-explorer-educators`
   - **Branch**: `main`
   - **File path**: `app.py`

4. **Click "Deploy"** — your app will be live in ~2 minutes!

Your app will be accessible at: `https://your-app-name.streamlit.app`

**Tip**: Share this URL with educators, students, and colleagues. No installation required—they just click the link!

### Deploy to Hugging Face Spaces (Alternative)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to **Spaces** → **+ New Space**
3. Set **SDK** to **Streamlit**, choose **Public**, select **Free CPU**
4. Upload `app.py` and `requirements.txt`
5. Hugging Face will deploy automatically

### Deploy to Render (More Control)

1. Create a GitHub repository
2. Go to [render.com](https://render.com) and create a **New Web Service**
3. Connect your GitHub repo
4. Set **Start Command** to `streamlit run app.py`
5. Choose **Free** tier and click **Create**

## Model Details

| Model | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| **distilgpt2** | 82M | Fast | Good | Demos, live classrooms |
| **gpt2** | 124M | Medium | Better | Detailed analysis |
| **bert-base-uncased** | 110M | Medium | Good | Masked language modeling |

*Note*: Models are downloaded on first run (~500 MB each) and cached locally for fast startup.

## Understanding the Output

### Probability Chart
- **X-axis**: Probability (0 to 1)
- **Y-axis**: Candidate tokens
- **Colors**: Green (high confidence), Yellow (medium), Red (low)
- **Blue highlight**: Token model actually chose

### Cumulative Probability Plot
- **Y-axis**: Log-scale probability (drops exponentially)
- **Trend**: Shows how "surprising" each generation step was
- **Insight**: Steep drops = model was uncertain; flat lines = confident predictions

### Token Table
- **Step**: Generation order (1, 2, 3, ...)
- **Token**: The actual word/token selected
- **Probability**: How likely the model thought this token was

## Technical Architecture

```
┌─────────────────────┐
│  User Input (Text)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Tokenizer (Hugging) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Model (Causal LM)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Softmax + Sampling │
│  (Temperature)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Visualization      │
│  (Matplotlib)       │
└─────────────────────┘
```

- **Transformers**: Pre-trained models from Hugging Face Model Hub
- **Streamlit**: Fast, responsive web interface (no JavaScript needed)
- **Matplotlib**: Probability distribution charts
- **PyTorch**: Efficient inference

## Extending the App

Want to customize or add features? Here are some ideas:

- **Add more models**: Edit `MODEL_OPTIONS` in `app.py`
- **Custom starting prompts**: Pre-load common examples for students
- **Leaderboard**: Track which tokens students guessed correctly
- **Export feature**: Save generated sequences as text
- **Multi-language support**: Add tokenizers for other languages
- **Advanced metrics**: Perplexity, entropy, divergence calculations

See `CONTRIBUTING.md` for development guidelines.

## Classroom Tips

📊 **For Live Demonstrations**
1. Use **distilgpt2** for speed (especially important for live demos)
2. Start with **low temperature** (0.1–0.3) to show deterministic behavior
3. Gradually increase temperature and show how outputs become creative/surprising
4. Use the **Guess Mode** to engage students actively

👥 **For Student Assignments**
- Ask students to predict tokens before clicking "Predict"
- Have them explain *why* they think a token is likely
- Discuss how different temperatures might be useful for different applications
  - Low temp: Technical writing, code generation
  - High temp: Creative writing, brainstorming

🎯 **Discussion Prompts**
- "Why do you think the model chose this token?"
- "What happens to probability as the sentence gets longer?"
- "How is this different from random word selection?"
- "Could this bias toward common tokens be a problem? Why or why not?"

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model takes too long to load** | Use distilgpt2 (smallest/fastest) or check internet connection |
| **Out of memory error** | Close other applications; ensure >2 GB RAM available |
| **Streamlit not found** | Run `pip install streamlit` in your virtual environment |
| **Transformers model not found** | Check internet; model will auto-download on first run |
| **Matplotlib plot not showing** | Refresh the browser; clear cache with `streamlit cache clear` |

## Citation

If you use **Token Explorer for Educators** in your teaching or research, please cite:

```bibtex
@software{token_explorer_2025,
  title={Token Explorer for Educators},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/token-explorer-educators}
}
```

## Credits / Attribution

This project builds upon code and ideas from [TKBEN Tokenizers Benchmarker](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker).
Original author: [CTCycle](https://github.com/CTCycle).

Certain components, logic, or inspiration were directly adapted or modified from this open-source repository.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Summary**: You're free to use, modify, and distribute this software for educational and commercial purposes.

## Contributing

We welcome contributions! Please:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

**Ideas for contributions**:
- New model options
- Improved UI/UX
- Translations
- Educational materials
- Bug fixes
- Performance optimizations

## Questions & Support

- **Issues**: Open a [GitHub Issue](https://github.com/YOUR_USERNAME/token-explorer-educators/issues)
- **Discussions**: Start a [GitHub Discussion](https://github.com/YOUR_USERNAME/token-explorer-educators/discussions)
- **Email**: YOUR_EMAIL@example.com

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) — Fast Python web app framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) — Pre-trained NLP models
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Matplotlib](https://matplotlib.org/) — Data visualization

## About

Created to democratize AI literacy and help educators teach how language models actually work.

**Last updated**: November 2025  
**Python version**: 3.8+  
**Streamlit version**: 1.0+

---

**Ready to deploy?** See [Deployment](#deployment) section above, or start locally with `streamlit run app.py`!
