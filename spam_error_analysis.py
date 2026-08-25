"""
Email Spam Classification - Error Analysis
Arch Technologies Internship - Month 1 - Task 1

Ye script wo exact messages dikhata hai jo model ne galat predict kiye,
taake pata chale ke error kyun ho raha hai aur kaise fix karein.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 1: Data aur model load karo
# ============================================================
df = pd.read_csv("spam_cleaned.csv")

with open("X_tfidf.pkl", "rb") as f:
    X = pickle.load(f)
with open("y_labels.pkl", "rb") as f:
    y = pickle.load(f)
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Same split jo training ke waqt use hua tha (random_state same rakhna zaroori hai)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Test set ke original text rows nikalne ke liye same indices chahiye
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# ============================================================
# STEP 2: Predictions lo aur galat wale nikalo
# ============================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

test_df = test_df.reset_index(drop=True)
test_df["predicted"] = y_pred
test_df["actual"] = y_test.reset_index(drop=True)
test_df["spam_probability"] = y_prob[:, 1] * 100

# Galat predictions
false_positives = test_df[(test_df["actual"] == 0) & (test_df["predicted"] == 1)]
false_negatives = test_df[(test_df["actual"] == 1) & (test_df["predicted"] == 0)]

print("=" * 60)
print(f"FALSE POSITIVES ({len(false_positives)}) - Ham ko galti se Spam kaha")
print("=" * 60)
for _, row in false_positives.iterrows():
    print(f"[Spam Prob: {row['spam_probability']:.1f}%] {row['text'][:80]}")
print()

print("=" * 60)
print(f"FALSE NEGATIVES ({len(false_negatives)}) - Spam ko galti se Ham kaha")
print("=" * 60)
for _, row in false_negatives.iterrows():
    print(f"[Spam Prob: {row['spam_probability']:.1f}%] {row['text'][:80]}")
print()

print("=" * 60)
print("OBSERVATION")
print("=" * 60)
print("In messages ko dekho - agar in mein spam-jaisi language kam hai")
print("(jaise koi 'free', 'win', 'urgent' jaisa word nahi), to ye normal")
print("errors hain jo har model mein hote hain - ambiguous/borderline cases.")
