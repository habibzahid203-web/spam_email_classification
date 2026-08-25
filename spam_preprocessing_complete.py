"""
Email Spam Classification - Complete Preprocessing Script
Arch Technologies Internship - Month 1 - Task 1

Ye script do kaam ek saath karta hai:
  PART 1: Raw dataset (spam.csv) ko clean karta hai
  PART 2: Cleaned data pe text-length outlier detection + boxplot banata hai

Output files:
  - spam_cleaned.csv     (cleaned dataset)
  - outlier_boxplot.png  (outlier visualization)
"""

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================================
# PART 1: DATA CLEANING
# ============================================================

# ---------- STEP 1: Dataset Load karo ----------
FILE_PATH = "email_data.csv"   # apni raw file ka naam/path yahan daal do

df = pd.read_csv(FILE_PATH, encoding='latin-1')

print("=" * 60)
print("STEP 1: RAW DATASET LOADED")
print("=" * 60)
print(f"Total Rows (raw): {df.shape[0]}")
print(f"Total Columns (raw): {df.shape[1]}")
print(f"Column Names: {list(df.columns)}")
print()

# ---------- STEP 2: Unnecessary/unnamed columns remove karo ----------
unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
if unnamed_cols:
    print(f"Removing {len(unnamed_cols)} unnamed/junk columns: {unnamed_cols}")
    df = df.drop(columns=unnamed_cols)

df.columns = [c.lower() for c in df.columns]
rename_map = {}
for col in df.columns:
    if col in ["v1", "category", "class", "type"]:
        rename_map[col] = "label"
    elif col in ["v2", "message", "text", "email", "content"]:
        rename_map[col] = "text"
df = df.rename(columns=rename_map)

print(f"Columns after standardizing: {list(df.columns)}")
print()

# ---------- STEP 3: Missing values check aur remove ----------
print("=" * 60)
print("STEP 2: MISSING VALUES CHECK")
print("=" * 60)
null_counts = df.isnull().sum()
print(null_counts)
total_nulls = null_counts.sum()

rows_before = df.shape[0]
df = df.dropna(subset=["label", "text"])
rows_after_null_removal = df.shape[0]

print(f"\nTotal missing values found: {total_nulls}")
print(f"Rows removed due to missing values: {rows_before - rows_after_null_removal}")
print()

# ---------- STEP 4: Duplicate rows check aur remove ----------
print("=" * 60)
print("STEP 3: DUPLICATE ROWS CHECK")
print("=" * 60)
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

rows_before_dup = df.shape[0]
df = df.drop_duplicates()
rows_after_dup = df.shape[0]

print(f"Rows removed due to duplicates: {rows_before_dup - rows_after_dup}")
print()

# ---------- STEP 5: Label column clean karo (spam/ham -> 1/0) ----------
print("=" * 60)
print("STEP 4: LABEL DISTRIBUTION (before encoding)")
print("=" * 60)
print(df["label"].value_counts())
print()

df["label"] = df["label"].astype(str).str.strip().str.lower()
df["label"] = df["label"].map({"spam": 1, "ham": 0})

rows_before_label_clean = df.shape[0]
df = df.dropna(subset=["label"])
rows_after_label_clean = df.shape[0]
print(f"Rows removed due to invalid/unexpected label values: {rows_before_label_clean - rows_after_label_clean}")
df["label"] = df["label"].astype(int)
print()

# ---------- STEP 6: Text cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')  # non-ascii/encoding artifacts remove (e.g. â£)
    text = re.sub(r'&lt;#&gt;|&lt;|&gt;', '', text)         # leftover HTML entity remnants (<#>, <, >)
    text = re.sub(r'\d+', ' numbertoken ', text)             # numbers ko delete karne ke bajaye token banao
                                                                # (phone/price/code spam ka strong signal hote hain)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    words = [w for w in words if w not in ("lt", "gt", "ltgt")]  # leftover fragments safety net
    return " ".join(words)

print("=" * 60)
print("STEP 5: TEXT CLEANING (lowercase, punctuation, numbers, stopwords)")
print("=" * 60)

sample_before = df["text"].iloc[0]
df["clean_text"] = df["text"].apply(clean_text)
sample_after = df["clean_text"].iloc[0]

print("Example - BEFORE cleaning:")
print(sample_before)
print("\nExample - AFTER cleaning:")
print(sample_after)
print()

rows_before_empty = df.shape[0]
df = df[df["clean_text"].str.strip() != ""]
rows_after_empty = df.shape[0]
print(f"Rows removed (became empty after cleaning): {rows_before_empty - rows_after_empty}")
print()

# ---------- CLEANING SUMMARY ----------
total_removed = rows_before - df.shape[0]
print("=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"Rows in RAW dataset            : {rows_before}")
print(f"Missing value rows removed     : {rows_before - rows_after_null_removal}")
print(f"Duplicate rows removed         : {rows_before_dup - rows_after_dup}")
print(f"Invalid label rows removed     : {rows_before_label_clean - rows_after_label_clean}")
print(f"Empty-after-cleaning removed   : {rows_before_empty - rows_after_empty}")
print(f"FINAL CLEAN dataset rows       : {df.shape[0]}")
print(f"Total removed                  : {total_removed} ({(total_removed/rows_before)*100:.2f}%)")
print()
print("Final Label Distribution:")
print(df["label"].value_counts().rename({0: "Ham (Not Spam)", 1: "Spam"}))
print()

# ---------- Cleaned file save karo ----------
df[["label", "text", "clean_text"]].to_csv("spam_cleaned.csv", index=False)
print("Cleaned dataset saved as: spam_cleaned.csv")
print()


# ============================================================
# PART 2: OUTLIER DETECTION
# ============================================================

print("=" * 60)
print("STEP 6: TEXT LENGTH STATISTICS")
print("=" * 60)

df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["clean_text"].apply(lambda x: len(str(x)))

print(df[["word_count", "char_count"]].describe())
print()

# ---------- IQR method se outliers detect karo ----------
Q1 = df["word_count"].quantile(0.25)
Q3 = df["word_count"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["word_count"] < lower_bound) | (df["word_count"] > upper_bound)]

print("=" * 60)
print("STEP 7: OUTLIER DETECTION (IQR Method)")
print("=" * 60)
print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print()
print(f"Total Outliers Found: {outliers.shape[0]}")
print(f"Outlier Percentage: {(outliers.shape[0] / df.shape[0]) * 100:.2f}%")
print()

if outliers.shape[0] > 0:
    print("Sample Outlier Rows (longest messages):")
    print(outliers[["label", "word_count", "text"]].sort_values("word_count", ascending=False).head(5))
    print()

# ---------- Boxplot banao ----------
plt.figure(figsize=(8, 5))
plt.boxplot(df["word_count"], vert=False, patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Word Count per Message (Outlier Detection)")
plt.xlabel("Word Count")
plt.tight_layout()
plt.savefig("outlier_boxplot.png", dpi=150)
print("Boxplot saved as: outlier_boxplot.png")
plt.show()

# ============================================================
# FINAL OVERALL SUMMARY
# ============================================================
print()
print("=" * 60)
print("FINAL OVERALL SUMMARY")
print("=" * 60)
print(f"Raw dataset rows        : {rows_before}")
print(f"Rows removed (cleaning) : {total_removed} ({(total_removed/rows_before)*100:.2f}%)")
print(f"Final clean dataset rows: {df.shape[0]}")
print(f"Outliers detected       : {outliers.shape[0]} ({(outliers.shape[0] / df.shape[0]) * 100:.2f}%)")
print(f"Normal word-count range : {lower_bound:.1f} to {upper_bound:.1f} words")
