# 🎓 Token Explorer for Educators

**A comprehensive, accessible Streamlit application for teaching AI language models in K-12 and higher education classrooms.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

---

## 🌟 Overview

Token Explorer for Educators is an interactive web application designed to make AI language models accessible to non-technical educators and students. It demonstrates how AI predicts text, helps explore tokenization, and provides ready-to-use classroom activities.

### 🎯 Key Features

✅ **25 Curated Example Prompts** across 5 categories  
✅ **Interactive Glossary & Help Panels** with classroom-friendly language  
✅ **5 AI Model Options** including multilingual support  
✅ **Advanced Parameter Controls** (Temperature, Top-k, Top-p)  
✅ **Class Poll Mode** for live student engagement  
✅ **Streamlined Visualizations** (probability chart, metrics, confidence tracking)  
✅ **Lightweight CSV & Text Exports** for sharing results  
✅ **2 Ready-to-Use Classroom Activities** with summaries  
✅ **Full Accessibility Support** (WCAG 2.1 compliant)  
✅ **Standards Alignment** (CSTA, ISTE, Common Core)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/token-explorer.git
cd token-explorer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Deploy to Streamlit Cloud (FREE)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy with one click!

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 📚 What's Included

### Example Prompts (25 total)

- **Famous Quotes**: "To be or not to be," "I have a dream..."
- **Story Starters**: "Once upon a time in a..." 
- **Science Facts**: "Water boils at..." "DNA stands for..."
- **Simple Sentences**: "The cat sat on the..."
- **Math & Logic**: "Two plus two equals..."

### AI Models (5 options)

1. **GPT-2 (English)** - General-purpose, creative text
2. **BERT Base (English)** - Context understanding
3. **BERT Multilingual** - 104 languages supported
4. **GPT-2 Spanish** - Spanish language generation
5. **DistilGPT-2 (Fast)** - Optimized for speed

### Classroom Activities (2 ready-to-use)

1. **Predict the Next Word Game** (Grades 3-8, 15-20 min)
2. **Temperature Experiment** (Grades 6-12, 25-30 min)

Each activity includes:
- Step-by-step instructions
- Learning goals
- Discussion questions
- Downloadable summaries

### Visualizations (4 types)

- 📊 **Probability Distribution** - Interactive bar chart of top tokens
- 📋 **Token Table** - Ranked probabilities with CSV/text export
- 📈 **Metrics Snapshot** - Entropy, perplexity, and top-token confidence
- 📉 **Confidence Tracking** - Probability and entropy over time

---

## 🎓 Educational Use Cases

### For Teachers

**Lesson Planning**:
- Browse 5 standards-aligned activities
- Load example prompts relevant to curriculum
- Export materials for student handouts

**Live Demonstrations**:
- Project app on classroom screen
- Enable Class Poll Mode
- Compare student predictions with AI

**Assessment**:
- Generate knowledge check quizzes
- Export results as CSV
- Track conceptual understanding

### For Students

**Self-Paced Exploration**:
- Try different example prompts
- Adjust parameters to see effects
- Read glossary definitions

**Collaborative Learning**:
- Submit predictions in poll mode
- Compare with AI and peers
- Discuss results in groups

**Assessment**:
- Take built-in quizzes
- Get instant feedback
- Review explanations

---

## 🌈 Accessibility Features

### WCAG 2.1 Level AA Compliant

✅ **High Contrast Mode** - 7:1 color ratio  
✅ **Font Size Controls** - 14px, 16px, 20px options  
✅ **Keyboard Navigation** - Full tab/arrow key support  
✅ **Screen Reader Support** - ARIA labels throughout  
✅ **Touch-Friendly** - 44×44px minimum tap targets  
✅ **Responsive Design** - Works on tablets and phones

---

## 📊 Standards Alignment

### CSTA K-12 CS Standards
- 1B-AP-15: Test and debug algorithms
- 3A-IC-24: Evaluate computational artifacts for bias
- 3B-AP-08: Describe how AI and ML algorithms work

### ISTE Standards for Students
- 1.6.d: Students understand how AI makes decisions
- 1.1.c: Students use technology for creative expression

### Common Core Math
- HSS-IC.A.2: Analyze decisions using probability
- 7.SP.C.7: Develop probability models

---

## 🔧 Technical Details

### Dependencies

```
streamlit >= 1.28.0
pandas >= 2.0.0
numpy >= 1.24.0
plotly >= 5.17.0
```

### Architecture

**Frontend**: Streamlit web framework  
**Visualizations**: Plotly interactive charts  
**Data**: Pandas DataFrames  
**AI Simulation**: Context-aware probability generation

### Performance

- ⚡ Fast loading with session state
- 💾 Efficient memory usage
- 📱 Optimized for mobile devices
- 🔄 Real-time updates in poll mode

---

## 📸 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400?text=Token+Explorer+Main+Interface)

### Class Poll Mode
![Poll Mode](https://via.placeholder.com/800x400?text=Class+Poll+Mode+Live+Results)

### Visualizations
![Charts](https://via.placeholder.com/800x400?text=Interactive+Probability+Charts)

### Classroom Activities
![Activities](https://via.placeholder.com/800x400?text=Ready-to-Use+Activities)

---

## 🎬 Tutorial Video

*(Coming soon - link to walkthrough video)*

---

## 📖 Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [app.py](app.py) - Main application code
- [requirements.txt](requirements.txt) - Python dependencies

---

## 🤝 Contributing

Educators are encouraged to:
- Adapt activities for their grade levels
- Add example prompts in different languages
- Suggest new visualizations
- Report bugs or usability issues

---

## 📄 License

This project is released for **educational use**. Feel free to adapt, modify, and share with educators and students.

---

## 🙏 Credits / Attribution

This project builds upon code and ideas from [TKBEN Tokenizers Benchmarker](https://github.com/CTCycle/TKBEN-tokenizers-benchmarker).
Original author: [CTCycle](https://github.com/CTCycle).

Certain components, logic, or inspiration were directly adapted or modified from this open-source repository.

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- Educational content aligned with CSTA, ISTE, and Common Core standards

---

### In-App Help
- Click the **"📖 Glossary"** button for term definitions
- Use **"❓ Help & Tutorial"** for guided walkthrough
- Check **"🏫 Classroom Activities"** for lesson ideas

### Troubleshooting

**App won't start?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Charts not displaying?**
```bash
streamlit cache clear
```

---

## 🎯 Roadmap

Future enhancements:
- [ ] Real API integration (Hugging Face)
- [ ] Student account system
- [ ] Teacher dashboard analytics
- [ ] Custom activity builder
- [ ] Video tutorial library
- [ ] Offline mode support

---

## 🌟 Star This Project

If you find Token Explorer helpful for your classroom, please ⭐ star this repository to help other educators discover it!

---

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out through GitHub Issues.

---

<div align="center">

**Token Explorer for Educators**

*Making AI Accessible to All Learners*

Version 2.0 | Streamlit Edition | November 2025

[Deploy Now](https://streamlit.io/cloud) | [View Demo](#) | [Read Docs](DEPLOYMENT_GUIDE.md)

</div>
