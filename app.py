import streamlit as st
import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration (No Sidebar by default)
st.set_page_config(page_title="Sarcasm Detector", page_icon="✨", layout="centered")

# 2. Modern Aesthetic CSS (Glassmorphism + Gradients)
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Hide standard Streamlit header and footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Titles */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

/* Metrics styles */
.metric-value {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

/* Text Area */
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 1px #8b5cf6 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: transparent;
    color: #f8fafc;
    border-bottom: 2px solid #8b5cf6;
}

/* Results */
.result-sarcastic {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: #fca5a5;
    margin-top: 20px;
}
.result-genuine {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: #86efac;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# 3. Model Loading & Training (High Accuracy Version)
@st.cache_data
def load_data():
    data = []
    try:
        with open("Sarcasm_Headlines_Dataset.json", "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

@st.cache_resource
def get_model(df):
    if df.empty: return None, None, 0.0
    
    X = df["headline"]
    y = df["is_sarcastic"]
    
    # We remove stop_words for sarcasm detection and add bigrams, which boosts accuracy to 85%+
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # High capacity Logistic Regression model
    model = LogisticRegression(C=5.0, max_iter=2000)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return vectorizer, model, acc

df = load_data()
vectorizer, model, acc = get_model(df)


# 4. Main App Layout
st.markdown('<div style="text-align: center; padding-bottom: 20px;"><h1 style="font-size: 3rem; background: linear-gradient(90deg, #c084fc, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Sarcasm Detection ML</h1><p style="color: #94a3b8; font-size: 1.1rem;">Machine Learning algorithm to classify intended sarcasm in text.</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Data Insights", "🧠 Word Cloud"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter a statement to analyze", placeholder="Type your sentence here... e.g. 'Oh, great, another Monday!'", height=120)
    
    if st.button("Analyze Sentence 🔍", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing linguistic patterns..."):
                time.sleep(0.5)
                vec = vectorizer.transform([user_input])
                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0][pred] * 100
                
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-sarcastic">
                        <h2 style="font-size: 2rem; margin:0;">😏 Sarcastic!</h2>
                        <p style="margin-bottom:0; opacity:0.9;">Confidence: <strong>{prob:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-genuine">
                        <h2 style="font-size: 2rem; margin:0;">😊 Genuine!</h2>
                        <p style="margin-bottom:0; opacity:0.9;">Confidence: <strong>{prob:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text first.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show ML Accuracy
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-label">Model Evaluation Accuracy</div>
        <div class="metric-value">{acc * 100:.2f}%</div>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 10px;">Calculated using Bigram TF-IDF + High-capacity Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Dataset Overview")
    st.markdown("<p style='color:#94a3b8'>We use the News Headlines Dataset for Sarcasm Detection containing thousands of samples.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Total Samples:** {len(df):,}")
    with col2:
        st.markdown(f"**Sarcastic Ratio:** {(df['is_sarcastic'].sum()/len(df))*100:.1f}%")
        
    st.dataframe(df[['headline', 'is_sarcastic']].head(50), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Frequent Word Associations")
    st.markdown("<p style='color:#94a3b8'>These are the most common words found in our dataset.</p>", unsafe_allow_html=True)
    if not df.empty:
        with st.spinner("Generating..."):
            text = " ".join(df["headline"].astype(str))
            bc = "#1e1b4b"
            wordcloud = WordCloud(width=800, height=400, background_color=bc, colormap="cool", max_words=100).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=bc)
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
