# Token Explorer for Educators - Streamlit Deployment Guide

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (public or private)
2. **Upload these files**:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (optional documentation)

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: https://streamlit.io/cloud
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select**:
   - Repository: `your-username/token-explorer`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click "Deploy"**

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📦 Alternative Deployment Options

### Option 1: Heroku

```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

### Option 2: Google Cloud Run

```bash
# Build and deploy
gcloud run deploy token-explorer \
  --source . \
  --platform managed \
  --region us-central1
```

### Option 3: AWS EC2

```bash
# On your EC2 instance
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
streamlit run app.py --server.port 80
```

### Option 4: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t token-explorer .
docker run -p 8501:8501 token-explorer
```

---

## 🔧 Configuration Options

### Custom Theme (optional)

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#0066CC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#212529"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

### Environment Variables

For production, set these environment variables:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 📊 Feature Highlights

1. **Guided Onboarding**
   - Quick start walkthrough
   - Interactive glossary with educator-friendly language

2. **Curated Examples**
   - 25 prompts across five classroom themes
   - Random prompt generator for instant variety

3. **Contextual Explanations**
   - Color-coded probability badges
   - Real-time entropy and perplexity indicators

4. **Model Options & Controls**
   - Five simulated model profiles (including multilingual)
   - Temperature, Top-k, and Top-p sliders with presets

5. **Class Poll Mode**
   - Collect student guesses anonymously
   - Human vs AI comparison table and grouped bar chart

6. **Visual Analytics**
   - Plotly probability chart
   - Ranked token table with CSV export
   - Confidence tracking timeline

7. **Lightweight Exports**
   - CSV download of token probabilities
   - Text summaries for predictions and activities

8. **Accessibility**
   - High-contrast mode
   - Adjustable font sizing
   - Large tap targets and ARIA-friendly markup

9. **Classroom Activities**
   - Two ready-to-use activity outlines
   - Downloadable text handouts for quick printing

10. **Standards Alignment**
    - CSTA K-12 CS Standards
    - ISTE Standards
    - Common Core Math

---

## 🎓 Usage in the Classroom

### For Teachers:

1. **Lesson Planning**:
   - Review classroom activities
   - Load example prompts
   - Download summaries as CSV or TXT

2. **Live Demonstrations**:
   - Enable Class Poll Mode
   - Project on classroom screen
   - Compare student vs. AI predictions

3. **Assessment & Reflection**:
   - Discuss entropy/perplexity trends
   - Encourage student reflection via downloads

### For Students:

1. **Self-Paced Exploration**:
   - Try different example prompts
   - Adjust parameters to see changes
   - Read glossary for definitions

2. **Collaborative Learning**:
   - Submit predictions in poll mode
   - Compare with classmates
   - Discuss AI behavior

3. **Assessment & Reflection**:
   - Review confidence tracking data
   - Compare human vs AI results
   - Reflect on prediction strategies

---

## 🔍 Troubleshooting

### Common Issues:

**1. Dependencies not installing**
```bash
# Upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

**2. Port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**3. Plotly charts not rendering**
```bash
# Clear Streamlit cache
streamlit cache clear
```

**4. High memory usage**
- Reduce the number of example prompts
- Limit prediction history
- Clear session state periodically

---

## 📚 Documentation

### Key Files:

- `app.py` - Main application (~40KB)
- `requirements.txt` - Python dependencies
- `README.md` - This deployment guide

### Technical Details:

**Simulated AI Features**:
- Context-aware probability generation
- Temperature-adjusted distributions
- Entropy and perplexity calculations
- Realistic tokenization

**Performance**:
- Streamlit session state for persistence
- Efficient data structures
- Lazy loading of visualizations

---

## 🌟 Best Practices

### For Classroom Use:

1. **Pre-load examples** before class
2. **Test your internet connection**
3. **Have backup plans** (downloaded CSV data)
4. **Use high contrast mode** for projection
5. **Increase font size** for visibility

### For Development:

1. **Keep dependencies minimal**
2. **Cache expensive computations**
3. **Handle errors gracefully**
4. **Add logging for debugging**
5. **Test on mobile devices**

---

## 🔒 Privacy & Security

- **No data collection**: All processing happens locally
- **No external APIs**: Simulated AI responses
- **No user accounts**: Anonymous usage
- **No personal information**: Poll mode is anonymous

---

## 📞 Support

For issues or questions:
1. Check the in-app glossary and help
2. Review classroom activities for guidance
3. Consult the tutorial walkthrough

---

## 🎯 Next Steps

After deployment:

1. **Test all features** with sample data
2. **Train teachers** on key functionalities
3. **Pilot in one classroom** before rolling out
4. **Gather feedback** from educators and students
5. **Iterate and improve** based on usage

---

## 📄 License

Educational use encouraged. Adapt for your classroom needs.

---

**Token Explorer for Educators** - Making AI Accessible to All Learners

Version 2.0 | Streamlit Edition | November 2025
