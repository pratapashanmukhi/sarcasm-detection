### 🎭 Sarcasm Detection ML – Interactive Text Classifier

Sarcasm Detection ML is a machine learning-based text classification tool built using Python, Scikit-learn, and Streamlit.
It reads user text or headlines, understands the linguistic context using NLP, and predicts whether the statement is sarcastic or genuine.
An NLP-based Sarcasm Detection dashboard achieving >85% accuracy using real-time ML classification and dynamic data visualizations.

## 📌 Project Description
Sarcasm Detection ML acts as a smart linguistic analyzer.
Users can input any sentence into the Streamlit dashboard, and the AI responds instantly with a prediction.
It provides real-time confidence scores (probabilities) and dynamically visualizes the dataset through insights and word clouds to help explain data trends.

## ✨ Features
🎙️ **Interactive Input** via a sleek Web UI
🤖 **Powered by Machine Learning** (Logistic Regression + Bigram TF-IDF)
📊 **Real-time Confidence Scoring** for predictions
🧠 **Dynamic Word Clouds** showing frequent linguistic associations
☁️ **Glassmorphism Aesthetic UI** built natively in Streamlit
✅ **High Accuracy Classification** (>85% baseline accuracy)

## 🛠️ Technologies Used
- Python 3
- Scikit-learn
- Streamlit
- Pandas
- Matplotlib
- WordCloud
- JSON (Dataset format)

## 📦 Required Python Libraries
Install these using pip:
- `streamlit`
- `pandas`
- `scikit-learn`
- `wordcloud`
- `matplotlib`

## ⚙️ Installation & Setup

### ✅ Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sarcasm-detection.git
cd sarcasm-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
Activate the virtual environment using the command below. 

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
If `requirements.txt` is not available:
```bash
pip install streamlit pandas scikit-learn wordcloud matplotlib
```

### 5. How to Run the Project
Navigate into the codes folder and run the Streamlit app:
```bash
cd codes
streamlit run app.py
```

## 🧑‍💻 How Sarcasm Detection ML Works
1. Loads the News Headlines Sarcasm Dataset from a local JSON file.
2. Converts raw text headlines into semantic vectors using TF-IDF (Term Frequency-Inverse Document Frequency) factoring in bigrams.
3. Passes the vectorized input through a high-capacity Logistic Regression classification model.
4. Predicts the outcome (1 for Sarcastic, 0 for Genuine) and calculates a probability confidence score.
5. Displays the result visually in an interactive, dark-themed glassmorphism card format.

## 🚀 Usage
Run the script:
```bash
streamlit run app.py
```
- Open the local URL provided by Streamlit in your web browser.
- Type any sentence into the "**Enter a statement to analyze**" box.
- Click **Analyze Sentence 🔍** and instantly see if your text was sarcastic or genuine.
- Explore the **Data Insights** and **Word Cloud** tabs to interact with the broader dataset.

## 🧠 Code Structure
- **`codes/app.py`** : Main Streamlit application containing the trained model pipeline, UI rendering, and data visualization generators.
- **`codes/run_accuracy.py`** : Script to evaluate baseline model accuracy in the terminal.
- **`Sarcasm_Headlines_Dataset.json`** : The raw JSON dataset of news headlines.

## 🤝 Contributing
Contributions are welcome! If you'd like to suggest improvements, report bugs, or add new features, please open an issue or submit a pull request.

## 📬 Contact
For queries or suggestions, please contact:
**Pratapa Shanmukhi**
🔗 **LinkedIn:** Pratapa Shanmukhi

## 🚀 Future Enhancements
- Support for API-based large language models (like Gemini)
- Deploy to Streamlit Cloud for public web access
- Expand dataset to include social media comments (X/Twitter, Reddit)# sa
