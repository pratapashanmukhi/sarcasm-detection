# NLP Sarcasm Detection System - Interactive Web Application

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-blue.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time Natural Language Processing (NLP) classification engine achieving >85% accuracy on news headlines and conversational text by analyzing linguistic irony, lexical polarity, and stylistic n-gram syntax patterns within a modern Streamlit web application.

> **Research Companion Repository:**  
> Looking for the complete machine learning benchmark suite with statistical cross-validation, radar plots, violin distributions, and automated technical report generation? Visit [`sarcasm-detection-ml`](https://github.com/pratapashanmukhi/sarcasm-detection-ml).

---

## Project Overview

Sarcasm presents a notorious obstacle in computational sentiment analysis because the literal semantic meaning of a statement contradicts the author's true intent (e.g., *"Oh brilliant, my flight was delayed for another six hours!"*). Standard bag-of-words pipelines frequently misclassify sarcastic complaints as positive sentiment.

This project delivers a production-grade NLP text mining pipeline that ingests headline corpora, applies text normalization, extracts TF-IDF n-gram vector representations (unigrams and bigrams), and deploys high-accuracy machine learning classifiers within a custom glassmorphism Streamlit analytics dashboard.

### Problem Statement
- **Polarity Inversion**: Sarcastic text utilizes overtly positive sentiment tokens within negative operational contexts, corrupting automated sentiment scoring.
- **Latency in Moderation**: High-throughput social media and customer feedback platforms require sub-100ms inference without heavy multi-gigabyte GPU models.
- **Explainability**: Black-box classifiers provide binary predictions without indicating which linguistic tokens drove the decision.

---

## Key Features

- **High-Accuracy Classification**: Achieves >85% classification accuracy on benchmark corpora (*The Onion* satirical headlines vs. *HuffPost* genuine news).
- **Glassmorphism Streamlit Dashboard**: Sleek UI featuring live sarcasm confidence gauges, phrase length analytics, and dynamic confidence intervals.
- **N-Gram Context Modeling**: Captures contextual phrasing via unigram + bigram TF-IDF vectorization without aggressive stopword deletion.
- **Word Cloud Analytics**: Visualizes distinctive lexical distributions between sarcastic and authentic news corpora.
- **Sub-50ms Inference Latency**: Optimized inference pipeline suitable for real-time customer review moderation.

---

## Architecture & Workflow

```
[ Raw Headline / Text Input ]
              │
              ▼
  [ NLTK Tokenization & Cleaning ]
   └── Lowercasing, Regex Stripping, Punctuation Preservation
              │
              ▼
  [ TF-IDF Vectorization ]
   └── (1, 2) N-Gram Feature Matrix (85%+ Accuracy)
              │
              ▼
  [ Trained Logistic Regression Core ]
   ├── Sarcastic vs Non-Sarcastic Probability
   └── Dynamic Decision Boundary Calibration
              │
              ▼
  [ Streamlit Interactive Interface (app.py) ]
   ├── Live Gauge Metric & Confidence Level
   └── Lexical Token & Word Cloud Insights
```

---

## Project Structure

```
sarcasm-detection/
├── app.py                          # Streamlit web application & inference engine
├── Sarcasm_Headlines_Dataset.json  # Benchmark annotated dataset (26,000+ headlines)
├── requirements.txt                # Pinned production dependencies
├── .gitignore                      # Environment, bytecode, and cache exclusions
└── README.md                       # Technical documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratapashanmukhi/sarcasm-detection.git
   cd sarcasm-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:8501`.

---

## Dataset Information

- **Dataset**: *News Headlines Dataset for Sarcasm Detection*
- **Volume**: 26,709 annotated news headlines
- **Sources**: Satirical articles (*The Onion*) vs Genuine news (*Huffington Post*)
- **Attributes**: `headline` (string), `is_sarcastic` (binary 0 or 1), `article_link` (URL)

---

## Future Improvements

- Transformer-based fine-tuning (RoBERTa / DeBERTa) for nuanced cultural sarcasm recognition.
- Multi-modal sentiment classification incorporating paired image thumbnails.
- Browser extension for live social media and forum comment scoring.

---

## Author

**Shanmukhi Pratapa**  
B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
VIT-AP University  
GitHub: [@pratapashanmukhi](https://github.com/pratapashanmukhi)
