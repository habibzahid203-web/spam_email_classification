"""
Email Spam Classification - Skewness Analysis
Arch Technologies Internship - Month 1 - Task 1

Ye script cleaned dataset (spam_cleaned.csv) ke numeric features
(word_count, char_count) ki skewness check karta hai aur
log-transformation se fix karne ka comparison bhi dikhata hai.

Output image:
  - skewness_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# ============================================================
# STEP 1: Dataset load karo aur numeric features banao
# ============================================================
df = pd.read_csv("spam_cleaned.csv")

df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["clean_text"].apply(lambda x: len(str(x)))

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(f"Total rows: {df.shape[0]}")
print()

# ============================================================
# STEP 2: Skewness calculate karo (before transformation)
# ============================================================
print("=" * 60)
print("STEP 2: SKEWNESS - BEFORE TRANSFORMATION")
print("=" * 60)

word_skew = skew(df["word_count"])
char_skew = skew(df["char_count"])

print(f"word_count skewness : {word_skew:.3f}")
print(f"char_count skewness : {char_skew:.3f}")
print()

def interpret_skew(value):
    if value > 1:
        return "Highly Right (Positively) Skewed"
    elif value > 0.5:
        return "Moderately Right Skewed"
    elif value < -1:
        return "Highly Left (Negatively) Skewed"
    elif value < -0.5:
        return "Moderately Left Skewed"
    else:
        return "Approximately Symmetric"

print(f"word_count interpretation : {interpret_skew(word_skew)}")
print(f"char_count interpretation : {interpret_skew(char_skew)}")
print()

# ============================================================
# STEP 3: Log Transformation apply karo (skewness fix karne ke liye)
# ============================================================
# log1p = log(1 + x), taake agar koi value 0 ho to bhi error na aaye
df["word_count_log"] = np.log1p(df["word_count"])
df["char_count_log"] = np.log1p(df["char_count"])

word_skew_log = skew(df["word_count_log"])
char_skew_log = skew(df["char_count_log"])

print("=" * 60)
print("STEP 3: SKEWNESS - AFTER LOG TRANSFORMATION")
print("=" * 60)
print(f"word_count_log skewness : {word_skew_log:.3f}")
print(f"char_count_log skewness : {char_skew_log:.3f}")
print()
print(f"word_count_log interpretation : {interpret_skew(word_skew_log)}")
print(f"char_count_log interpretation : {interpret_skew(char_skew_log)}")
print()

# ============================================================
# STEP 4: Before vs After ka visual comparison (histograms)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(df["word_count"], bins=30, color="#F44336", edgecolor="black")
axes[0, 0].set_title(f"word_count - BEFORE (Skew={word_skew:.2f})")
axes[0, 0].set_xlabel("Word Count")

axes[0, 1].hist(df["word_count_log"], bins=30, color="#4CAF50", edgecolor="black")
axes[0, 1].set_title(f"word_count - AFTER Log Transform (Skew={word_skew_log:.2f})")
axes[0, 1].set_xlabel("Log(1 + Word Count)")

axes[1, 0].hist(df["char_count"], bins=30, color="#F44336", edgecolor="black")
axes[1, 0].set_title(f"char_count - BEFORE (Skew={char_skew:.2f})")
axes[1, 0].set_xlabel("Character Count")

axes[1, 1].hist(df["char_count_log"], bins=30, color="#4CAF50", edgecolor="black")
axes[1, 1].set_title(f"char_count - AFTER Log Transform (Skew={char_skew_log:.2f})")
axes[1, 1].set_xlabel("Log(1 + Character Count)")

plt.tight_layout()
plt.savefig("skewness_analysis.png", dpi=150)
print("Saved: skewness_analysis.png")
plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"word_count : {word_skew:.3f} -> {word_skew_log:.3f} (after log transform)")
print(f"char_count : {char_skew:.3f} -> {char_skew_log:.3f} (after log transform)")
