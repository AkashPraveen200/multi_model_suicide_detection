# Multi-Model Suicide Detection System

A multimodal artificial intelligence system designed to evaluate suicidal ideation risk by analyzing text, voice, and video inputs.

## 📌 Features

- **Text Analysis**: Fine-tuned ELECTRA Transformer model trained for sequence classification of suicide ideation in text and chat transcripts.
- **Voice Analysis**: Automated Speech-to-Text transcription via Google SpeechRecognition followed by ELECTRA sentiment and risk evaluation.
- **Video Emotion Detection**: Facial emotion analysis powered by DeepFace and PyTorch to classify emotional states (sad, angry, fear, disgust, neutral, happy, surprise).
- **Multimodal Fusion**: Dynamically weighted probability fusion combining text (40%), voice (40%), and video (20%) modalities.
- **Streamlit Dashboard**: Interactive web application for daily input testing, real-time multimodal analysis, and automated weekly report generation.

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AkashPraveen200/multi_model_suicide_detection.git
   cd multi_model_suicide_detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Command-Line Prediction Pipeline
```bash
python predict.py
```

### Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to access the dashboard.
