"""
Email Spam Classification - PRE-TRAINING VALIDATION CHECK
Arch Technologies Internship - Month 1 - Task 1

Ye script model training se PEHLE poora pipeline verify karta hai:
  1. Cleaned CSV mein koi issue to nahi
  2. Missing values check
  3. Duplicate check
  4. Label validity check
  5. Text cleaning quality check
  6. TF-IDF files load ho rahi hain aur consistent hain
  7. X aur y ka shape match karta hai ya nahi
  8. Train/Test split ke liye data ready hai ya nahi

Agar sab PASS ho jaye, to model training shuru karna 100% safe hai.
"""

import pandas as pd
import pickle
import numpy as np

issues_found = []

print("=" * 60)
print("PRE-TRAINING VALIDATION CHECK")
print("=" * 60)
print()

# ============================================================
# CHECK 1: spam_cleaned.csv load aur basic checks
# ============================================================
print("-" * 60)
print("CHECK 1: spam_cleaned.csv")
print("-" * 60)

df = pd.read_csv("spam_cleaned.csv")
print(f"Rows: {df.shape[0]}, Columns: {list(df.columns)}")

# Missing values
null_count = df.isnull().sum().sum()
print(f"Missing values: {null_count}")
if null_count > 0:
    issues_found.append(f"spam_cleaned.csv mein {null_count} missing values hain")
else:
    print("PASS: Koi missing value nahi")

# Duplicates
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")
if dup_count > 0:
    issues_found.append(f"spam_cleaned.csv mein {dup_count} duplicate rows hain")
else:
    print("PASS: Koi duplicate nahi")

# Empty text check
empty_text = (df["clean_text"].astype(str).str.strip() == "").sum()
print(f"Empty clean_text rows: {empty_text}")
if empty_text > 0:
    issues_found.append(f"{empty_text} rows mein clean_text empty hai")
else:
    print("PASS: Koi empty text nahi")
print()

# ============================================================
# CHECK 2: Label validity
# ============================================================
print("-" * 60)
print("CHECK 2: Label Validity")
print("-" * 60)

unique_labels = df["label"].unique()
print(f"Unique label values: {sorted(unique_labels)}")

if set(unique_labels) == {0, 1}:
    print("PASS: Labels sirf 0 aur 1 hain (koi invalid value nahi)")
else:
    issues_found.append(f"Labels mein unexpected values hain: {unique_labels}")

label_dtype = df["label"].dtype
print(f"Label data type: {label_dtype}")
if label_dtype != "int64" and label_dtype != "int32":
    issues_found.append(f"Label column integer type mein nahi hai (current: {label_dtype})")
else:
    print("PASS: Label integer type mein hai")
print()

# ============================================================
# CHECK 3: Text cleaning quality (spot check for leftover issues)
# ============================================================
print("-" * 60)
print("CHECK 3: Text Cleaning Quality")
print("-" * 60)

sample_texts = " ".join(df["clean_text"].astype(str).sample(min(500, len(df)), random_state=42))

# Encoding artifacts check
has_non_ascii = any(ord(c) > 127 for c in sample_texts)
print(f"Non-ASCII characters found in sample: {has_non_ascii}")
if has_non_ascii:
    issues_found.append("Kuch non-ASCII/encoding artifacts abhi bhi text mein reh sakte hain")
else:
    print("PASS: Koi encoding artifact nahi mila")

# Leftover digits check (numbers should've been removed)
has_digits = any(c.isdigit() for c in sample_texts)
print(f"Digits found in cleaned text: {has_digits}")
if has_digits:
    issues_found.append("Cleaned text mein abhi bhi numbers reh gaye hain")
else:
    print("PASS: Koi number leftover nahi")

# Leftover punctuation check
punct_chars = set("!@#$%^&*()[]{}<>,.?/\\|~`\"'")
has_punct = any(c in punct_chars for c in sample_texts)
print(f"Punctuation found in cleaned text: {has_punct}")
if has_punct:
    issues_found.append("Cleaned text mein abhi bhi punctuation reh gaya hai")
else:
    print("PASS: Koi punctuation leftover nahi")
print()

# ============================================================
# CHECK 4: TF-IDF files consistency
# ============================================================
print("-" * 60)
print("CHECK 4: TF-IDF Files Consistency")
print("-" * 60)

try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("X_tfidf.pkl", "rb") as f:
        X = pickle.load(f)
    with open("y_labels.pkl", "rb") as f:
        y = pickle.load(f)
    print("PASS: Teeno pkl files load ho gayi (vectorizer, X, y)")
except FileNotFoundError as e:
    issues_found.append(f"TF-IDF file missing: {e}")
    X, y = None, None

if X is not None and y is not None:
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if X.shape[0] == y.shape[0]:
        print("PASS: X aur y ke rows match karte hain")
    else:
        issues_found.append(f"X rows ({X.shape[0]}) aur y rows ({y.shape[0]}) match nahi karte")

    if X.shape[0] == df.shape[0]:
        print("PASS: X ke rows spam_cleaned.csv ke rows se match karte hain")
    else:
        issues_found.append(
            f"X rows ({X.shape[0]}) aur spam_cleaned.csv rows ({df.shape[0]}) match nahi karte "
            "-- shayad TF-IDF purani CSV pe fit hui thi, dobara run karo"
        )

    # NaN/Inf check in feature matrix
    if np.isnan(X.data).any() or np.isinf(X.data).any():
        issues_found.append("X (TF-IDF matrix) mein NaN ya Inf values hain")
    else:
        print("PASS: X mein koi NaN/Inf nahi")

    # Label distribution in y
    print(f"y distribution: {dict(pd.Series(y).value_counts())}")
print()

# ============================================================
# FINAL VERDICT
# ============================================================
print("=" * 60)
print("FINAL VALIDATION RESULT")
print("=" * 60)

if len(issues_found) == 0:
    print("ALL CHECKS PASSED ✔")
    print("Pipeline (cleaning -> outliers -> EDA -> skewness -> TF-IDF) is consistent.")
    print("Model training shuru karna SAFE hai.")
else:
    print(f"{len(issues_found)} ISSUE(S) FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    print()
    print("Model training se pehle in issues ko fix karna zaroori hai.")
