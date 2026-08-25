"""
Email/SMS Spam Classification - Streamlit Web App
Arch Technologies Internship - Month 1 - Task 1

Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Spam Email Detection | Arch Technologies",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #F0FDFA 0%, #ECFEFF 50%, #F0F9FF 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #CCFBF1 0%, #E0F2FE 100%);
    }
    .app-header {
        background: linear-gradient(135deg, #0D9488, #0891B2);
        border-radius: 16px;
        padding: 1.6rem 1.2rem;
        text-align: center;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.25);
    }
    .app-header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
    }
    .app-header-subtitle {
        color: #D1FAE5;
        font-size: 0.98rem;
        margin-top: 0.35rem;
    }
    .result-box-spam {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box-ham {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label-spam {
        font-size: 1.8rem;
        font-weight: 800;
        color: #b91c1c;
    }
    .result-label-ham {
        font-size: 1.8rem;
        font-weight: 800;
        color: #15803d;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0D9488, #0891B2);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0F766E, #0E7490);
    }
    .example-btn button {
        background-color: #f3f4f6 !important;
        color: #0D9488 !important;
        border: 1px solid #d1d5db !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(13, 148, 136, 0.15);
    }
    .sidebar-card-title {
        font-weight: 700;
        color: #0F766E;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .sidebar-card-text {
        font-size: 0.85rem;
        color: #374151;
        line-height: 1.5;
    }
    .tip-item {
        font-size: 0.83rem;
        color: #374151;
        margin-bottom: 0.3rem;
    }
    .footer-badge {
        text-align: center;
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }
    [data-testid="stMetricValue"] {
        color: #0F766E;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TEXT CLEANING (same logic as training pipeline)
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'&lt;#&gt;|&lt;|&gt;', '', text)
    text = re.sub(r'\d+', ' numbertoken ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    words = [w for w in words if w not in ("lt", "gt", "ltgt")]
    return " ".join(words)


@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model


def predict_email(raw_text, tfidf, model):
    cleaned = clean_text(raw_text)
    if cleaned.strip() == "":
        return "Ham", 0.0, 100.0
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    label = "Spam" if prediction == 1 else "Ham"
    return label, probabilities[1] * 100, probabilities[0] * 100


# ============================================================
# LOAD MODEL
# ============================================================
try:
    tfidf, model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📧 Spam Detection")
    st.caption("Arch Technologies — ML Internship, Month 1, Task 1")
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">📖 About</div>
        <div class="sidebar-card-text">
            A Logistic Regression model trained on the SMS Spam
            Collection dataset with TF-IDF feature extraction,
            classifying messages as Spam or Ham.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">⚙️ Pipeline</div>
        <div class="sidebar-card-text">
            Cleaning → Outlier & Skewness Analysis<br>
            TF-IDF (unigrams + bigrams)<br>
            Logistic Regression (class-balanced)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">🗂️ Dataset</div>
        <div class="sidebar-card-text">
            SMS Spam Collection (Kaggle/UCI)<br>
            5,572 raw messages → 5,144 cleaned<br>
            87% Ham / 13% Spam
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "97.76%")
    col2.metric("F1-Score", "91.12%")
    col1.metric("Precision", "90.08%")
    col2.metric("Recall", "92.19%")

    with st.expander("🔍 Error Analysis Insight"):
        st.markdown("""
        Replacing digits with a **numbertoken** placeholder (instead of
        deleting them) preserved a key spam signal — phone numbers
        and prices — improving recall from 91.4% to 92.2%.
        """)

    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    <div class="tip-item">✉️ Paste the full message for best accuracy</div>
    <div class="tip-item">🔢 Numbers/prices help signal spam</div>
    <div class="tip-item">⚠️ Model is tuned for promotional spam, not social-engineering scams</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="footer-badge">Built with scikit-learn · TF-IDF · Streamlit</div>', unsafe_allow_html=True)

# ============================================================
# MAIN PAGE
# ============================================================
st.markdown("""
    <div class="app-header">
        <p class="app-header-title">📧 Spam Email Detection</p>
        <p class="app-header-subtitle">Paste any email or SMS text below to check if it's Spam or Ham</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "⚠️ Model files not found! Make sure `tfidf_vectorizer.pkl` and `spam_model.pkl` "
        "are in the same folder as this app, then restart."
    )
    st.stop()

# Example buttons
st.markdown("**Try an example:**")
ex_col1, ex_col2 = st.columns(2)

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

with ex_col1:
    if st.button("📩 Load Spam Example", use_container_width=True):
        st.session_state.message_input = (
            "Congratulations! You have won a FREE prize worth $1000. "
            "Call 09061701461 now to claim your reward before it expires. "
            "Text WIN to 80488 urgent!"
        )

with ex_col2:
    if st.button("💬 Load Ham Example", use_container_width=True):
        st.session_state.message_input = (
            "Hey, are you free tonight? Let's grab dinner around 8pm, "
            "I'll pick you up from your place. Let me know if that works for you."
        )

# Text input
message = st.text_area(
    "Message text:",
    value=st.session_state.message_input,
    height=150,
    placeholder="Type or paste an email/SMS message here...",
    key="message_input",
)

# Predict button
if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        label, spam_prob, ham_prob = predict_email(message, tfidf, model)

        if label == "Spam":
            st.markdown(f"""
                <div class="result-box-spam">
                    <div class="result-label-spam">🚨 SPAM DETECTED</div>
                    <p style="margin-top:0.5rem;color:#7f1d1d;">
                        Spam Probability: <b>{spam_prob:.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box-ham">
                    <div class="result-label-ham">✅ LOOKS SAFE (HAM)</div>
                    <p style="margin-top:0.5rem;color:#14532d;">
                        Ham Probability: <b>{ham_prob:.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Confidence Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Ham")
            st.progress(int(ham_prob))
            st.write(f"{ham_prob:.2f}%")
        with c2:
            st.write("Spam")
            st.progress(int(spam_prob))
            st.write(f"{spam_prob:.2f}%")

st.markdown("---")
st.caption("Built with Logistic Regression + TF-IDF | SMS Spam Collection Dataset")
