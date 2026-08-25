"""
Email Spam Classification - TF-IDF Feature Extraction
Arch Technologies Internship - Month 1 - Task 1

Ye script cleaned text (clean_text column) ko TF-IDF vectors mein convert karta hai
taake ML model train karne ke liye numeric input mil sake.

Output:
  - tfidf_vectorizer.pkl   (fitted vectorizer, baad mein reuse ke liye)
  - X_tfidf.pkl / y.pkl    (features aur labels, model training step ke liye)
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# STEP 1: Cleaned dataset load karo
# ============================================================
df = pd.read_csv("spam_cleaned.csv")

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(f"Total rows: {df.shape[0]}")
print()

# ============================================================
# STEP 2: TF-IDF Vectorizer banao aur fit karo
# ============================================================
# max_features: sirf top 3000 sabse important words rakhega (vocabulary size control)
# ngram_range=(1,2): single words + do-word phrases dono capture karega (jaise "free prize")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

print("=" * 60)
print("STEP 2: TF-IDF TRANSFORMATION DONE")
print("=" * 60)
print(f"Feature matrix shape: {X.shape}")
print(f"  -> {X.shape[0]} messages, {X.shape[1]} features (words/phrases)")
print()

# ============================================================
# STEP 3: Sample - kaunse words TF-IDF ne important samjhe
# ============================================================
feature_names = tfidf.get_feature_names_out()

print("=" * 60)
print("STEP 3: SAMPLE VOCABULARY (first 20 features)")
print("=" * 60)
print(list(feature_names[:20]))
print()

# Ek single message ka example - uske top TF-IDF scoring words
sample_idx = 0
sample_vector = X[sample_idx].toarray()[0]
top_indices = sample_vector.argsort()[::-1][:5]

print(f"Example message: {df['clean_text'].iloc[sample_idx]}")
print("Top 5 TF-IDF words for this message:")
for idx in top_indices:
    if sample_vector[idx] > 0:
        print(f"  {feature_names[idx]}: {sample_vector[idx]:.4f}")
print()

# ============================================================
# STEP 4: Save karo (agla step - model training - isko use karega)
# ============================================================
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("X_tfidf.pkl", "wb") as f:
    pickle.dump(X, f)

with open("y_labels.pkl", "wb") as f:
    pickle.dump(y, f)

print("=" * 60)
print("FILES SAVED")
print("=" * 60)
print("tfidf_vectorizer.pkl  -> fitted TF-IDF vectorizer")
print("X_tfidf.pkl            -> feature matrix (model training ke liye)")
print("y_labels.pkl            -> labels (spam/ham)")
print()
print("Ye files agle step (model training) mein directly load hongi.")
