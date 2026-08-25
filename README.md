# 📧 Spam Email Detection

A machine learning pipeline that classifies email/SMS messages as **Spam** or **Ham (not spam)**, built end-to-end — from raw data cleaning to a deployed, interactive web app.

Built as part of the **Machine Learning Internship at Arch Technologies** (Month 1, Task 1).

---

## 🚀 Live Demo

Run the Streamlit app locally to try it yourself — draw from real examples or paste your own message and get an instant Spam/Ham prediction with confidence scores.

---

## 📊 Overview

| | |
|---|---|
| **Task** | Binary text classification (Spam vs. Ham) |
| **Dataset** | SMS Spam Collection (Kaggle / UCI) — 5,572 raw messages |
| **Final Dataset** | 5,144 messages after cleaning |
| **Model** | Logistic Regression (class-balanced) |
| **Features** | TF-IDF (unigrams + bigrams, 3,000 features) |
| **Accuracy** | **97.76%** |
| **Precision / Recall / F1** | 90.08% / 92.19% / 91.12% |

---

## 🔧 Pipeline

1. **Data Cleaning** — removed duplicates, standardized labels, normalized text (lowercasing, URL/punctuation removal, stopword removal), fixed encoding artifacts
2. **Outlier Detection** — IQR method on message word-count; retained genuine long-form messages
3. **Exploratory Data Analysis** — class distribution, message-length comparison, most frequent spam/ham vocabulary
4. **Skewness Analysis** — log-transformation applied to normalize word/character count distributions
5. **Feature Engineering** — TF-IDF vectorization (unigrams + bigrams)
6. **Model Training** — Logistic Regression with `class_weight='balanced'` to handle class imbalance (87% Ham / 13% Spam)
7. **Error Analysis** — reviewed every misclassified message; found that stripping numeric values (phone numbers, prices) removed a key spam signal — fixed by replacing digits with a `numbertoken` placeholder instead of deleting them, improving recall from 91.4% → 92.2%
8. **Deployment** — interactive Streamlit web app + terminal-based prediction script

---

## 📁 Project Structure

```
spam-email-detection/
├── spam_preprocessing_complete.py   # Data cleaning, outlier detection
├── spam_eda.py                       # Exploratory data analysis + visualizations
├── spam_skewness.py                  # Skewness analysis + log transformation
├── spam_tfidf.py                     # TF-IDF feature extraction
├── spam_model_training.py            # Logistic Regression training + evaluation
├── spam_error_analysis.py            # Misclassification review
├── spam_validation_check.py          # Pre-training pipeline sanity checks
├── main.py                           # Terminal-based prediction script
├── app.py                            # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `Streamlit`

---

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<habibzahid203-web>/spam-email-detection.git
cd spam-email-detection

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (in order)
python spam_preprocessing_complete.py
python spam_tfidf.py
python spam_model_training.py

# Try it out
python main.py                  # terminal version
streamlit run app.py            # web app version
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | 97.76% |
| Precision (Spam) | 90.08% |
| Recall (Spam) | 92.19% |
| F1-Score | 91.12% |

**Confusion Matrix (test set, n=1029):**

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 888 | 13 |
| **Actual Spam** | 10 | 118 |

---

## 🔍 Key Learning

The biggest accuracy gain didn't come from a fancier model — it came from **error analysis**: manually reading misclassified messages revealed that deleting digits during cleaning was destroying a real spam signal (phone numbers, prices). Replacing digits with a placeholder token instead of removing them improved recall without any model changes.

---

## ⚠️ Known Limitation

The model is trained on a dataset of promotional/marketing-style spam and does not generalize well to social-engineering scam messages (e.g., refund/overpayment scams) that lack typical spam vocabulary like "free" or "prize." This is a direction for future improvement.

---

## 👤 Author

**Muhammad Habib Ur Rehman**
BSCS Student, University of Narowal | ML Intern, Arch Technologies

---

## 📄 License

This project is for educational purposes as part of an internship submission.
