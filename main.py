"""
Email Spam Classification - Prediction Script (main.py)
Arch Technologies Internship - Month 1 - Task 1

Ye script trained model (spam_model.pkl) aur TF-IDF vectorizer
(tfidf_vectorizer.pkl) load karke, tumhare diye gaye kisi bhi
email/message ko Spam ya Ham predict karta hai.

Usage:
    python main.py
    (phir terminal mein apna email/message type karo)
"""

import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================================
# Same cleaning function jo training ke waqt use hui thi
# (IMPORTANT: prediction ke waqt bhi exact wahi cleaning honi chahiye)
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


# ============================================================
# STEP 1: Trained model aur vectorizer load karo
# ============================================================
print("Loading trained model...")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!\n")


# ============================================================
# STEP 2: Prediction function
# ============================================================
def predict_email(raw_text):
    cleaned = clean_text(raw_text)

    if cleaned.strip() == "":
        return "Ham", 0.0, 0.0  # empty/meaningless text -> default Ham

    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    label = "Spam" if prediction == 1 else "Ham"
    ham_prob = probabilities[0] * 100
    spam_prob = probabilities[1] * 100

    return label, ham_prob, spam_prob


# ============================================================
# STEP 3: Interactive loop (terminal se test karo)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMAIL / SMS SPAM DETECTOR")
    print("=" * 60)
    print("Apna email/message type karo aur Enter dabao.")
    print("Band karne ke liye 'exit' type karo.\n")

    while True:
        user_input = input("Enter email/message: ").strip()

        if user_input.lower() == "exit":
            print("Program band ho raha hai...")
            break

        if user_input == "":
            print("Khali message hai, kuch likho.\n")
            continue

        label, ham_prob, spam_prob = predict_email(user_input)

        print("-" * 60)
        print(f"Prediction     : {label}")
        print(f"Ham Probability : {ham_prob:.2f}%")
        print(f"Spam Probability: {spam_prob:.2f}%")
        print("-" * 60)
        print()
