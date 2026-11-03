# NLP Sarcasm Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-blue.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time Natural Language Processing (NLP) classification engine achieving >85% accuracy on news headlines and informal text by analyzing linguistic irony, lexical polarity, and stylistic syntax patterns.

---

## Overview

Sarcasm presents a notorious challenge for standard computational sentiment analysis because the literal semantic meaning of a statement directly contradicts the speaker's true intent (e.g., *"Oh fantastic, my flight got delayed for another five hours!"*).

This project implements a complete NLP text mining pipeline that ingests headline corpora, applies text normalization (tokenization, lemmatization, stop-word pruning), computes TF-IDF n-gram vector representations, and deploys high-accuracy machine learning classifiers within an interactive Streamlit analytics dashboard.

---

## Key Features

- **High-Accuracy Classification**: Achieves >85% classification accuracy on benchmark sarcasm corpora (*The Onion* satirical headlines vs. *HuffPost* genuine news).
- **Interactive Sentiment Dashboard**: Web interface for real-time phrase evaluation, live sarcasm probability gauges, and dynamic confidence intervals.
- **Explainable Feature Importance**: Highlights linguistic tokens and n-gram patterns that contributed most heavily to the sarcasm prediction.
- **Lightweight & Fast**: Sub-50ms inference latency per query, making it well-suited for high-throughput social media and customer review moderation pipelines.

---

## Machine Learning Pipeline

```
[Raw Text Input] ────────> [Regex Cleaning & Lowercasing]
                                     │
                                     ▼
                           [NLTK Tokenization & Stopwords]
                                     │
                                     ▼
                           [TF-IDF Vectorizer (Uni/Bi-grams)]
                                     │
                                     ▼
                           [ML Classifier Ensemble]
                             ├── Logistic Regression
                             └── Linear Support Vector Machine
                                     │
                                     ▼
                           [Streamlit Interactive Output]
                             ├── Sarcastic / Non-Sarcastic Label
                             └── Confidence Probability Metric
```

---

## Project Structure

```
sarcasm-detection/
├── app.py                          # Streamlit web application & inference engine
├── Sarcasm_Headlines_Dataset.json  # Benchmark annotated dataset (26,000+ headlines)
├── requirements.txt                # Production dependencies
├── .gitignore                      # Exclusion rules
└── README.md                       # Technical documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip package manager

### Execution Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratapashanmukhi/sarcasm-detection.git
   cd sarcasm-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   Navigate to `http://localhost:8501`.

---

## Author

**Shanmukhi Pratapa**  
B.Tech CSE (AI & ML) — VIT-AP University  
GitHub: [@pratapashanmukhi](https://github.com/pratapashanmukhi)
