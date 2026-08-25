"""
Email Spam Classification - Exploratory Data Analysis (EDA)
Arch Technologies Internship - Month 1 - Task 1

Ye script cleaned dataset (spam_cleaned.csv) pe EDA karta hai:
  1. Spam vs Ham distribution (bar chart + pie chart)
  2. Message length comparison - spam vs ham (boxplot)
  3. Most common words - spam vs ham (bar charts)

Output images:
  - eda_class_distribution.png
  - eda_length_comparison.png
  - eda_common_words_spam.png
  - eda_common_words_ham.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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
# STEP 2: Spam vs Ham Distribution
# ============================================================
print("=" * 60)
print("STEP 2: CLASS DISTRIBUTION")
print("=" * 60)
counts = df["label"].value_counts().rename({0: "Ham", 1: "Spam"})
percentages = df["label"].value_counts(normalize=True).rename({0: "Ham", 1: "Spam"}) * 100

print(counts)
print()
print("Percentages:")
print(percentages.round(2))
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
axes[0].bar(counts.index, counts.values, color=["#4CAF50", "#F44336"])
axes[0].set_title("Spam vs Ham - Count")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Number of Messages")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"], startangle=90)
axes[1].set_title("Spam vs Ham - Percentage")

plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
print("Saved: eda_class_distribution.png")
plt.show()
print()

print("NOTE: Dataset imbalanced hai (Ham >> Spam) - model training ke waqt")
print("iska khayal rakhna hoga (jaise class_weight='balanced' use karna).")
print()

# ============================================================
# STEP 3: Message Length Comparison (Spam vs Ham)
# ============================================================
print("=" * 60)
print("STEP 3: MESSAGE LENGTH - SPAM vs HAM")
print("=" * 60)

df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))

spam_lengths = df[df["label"] == 1]["word_count"]
ham_lengths = df[df["label"] == 0]["word_count"]

print("Spam messages - avg word count:", round(spam_lengths.mean(), 2))
print("Ham messages  - avg word count:", round(ham_lengths.mean(), 2))
print()

plt.figure(figsize=(8, 5))
plt.boxplot([ham_lengths, spam_lengths], tick_labels=["Ham", "Spam"], patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title("Message Length Comparison: Spam vs Ham")
plt.ylabel("Word Count")
plt.tight_layout()
plt.savefig("eda_length_comparison.png", dpi=150)
print("Saved: eda_length_comparison.png")
plt.show()
print()

# ============================================================
# STEP 4: Most Common Words - Spam vs Ham
# ============================================================
print("=" * 60)
print("STEP 4: MOST COMMON WORDS (Top 15)")
print("=" * 60)

def get_top_words(texts, n=15):
    all_words = " ".join(texts).split()
    return Counter(all_words).most_common(n)

# Spam
spam_words = get_top_words(df[df["label"] == 1]["clean_text"])
print("Top 15 words in SPAM messages:")
for word, freq in spam_words:
    print(f"  {word}: {freq}")
print()

plt.figure(figsize=(8, 6))
words, freqs = zip(*spam_words)
plt.barh(words[::-1], freqs[::-1], color="#F44336")
plt.title("Top 15 Most Common Words in SPAM Messages")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("eda_common_words_spam.png", dpi=150)
print("Saved: eda_common_words_spam.png")
plt.show()
print()

# Ham
ham_words = get_top_words(df[df["label"] == 0]["clean_text"])
print("Top 15 words in HAM messages:")
for word, freq in ham_words:
    print(f"  {word}: {freq}")
print()

plt.figure(figsize=(8, 6))
words, freqs = zip(*ham_words)
plt.barh(words[::-1], freqs[::-1], color="#4CAF50")
plt.title("Top 15 Most Common Words in HAM Messages")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("eda_common_words_ham.png", dpi=150)
print("Saved: eda_common_words_ham.png")
plt.show()
print()

# ============================================================
# FINAL EDA SUMMARY
# ============================================================
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"Total messages          : {df.shape[0]}")
print(f"Spam                    : {counts.get('Spam', 0)} ({percentages.get('Spam', 0):.2f}%)")
print(f"Ham                     : {counts.get('Ham', 0)} ({percentages.get('Ham', 0):.2f}%)")
print(f"Avg word count (Spam)   : {round(spam_lengths.mean(), 2)}")
print(f"Avg word count (Ham)    : {round(ham_lengths.mean(), 2)}")
print(f"Top Spam word           : {spam_words[0][0]} ({spam_words[0][1]} times)")
print(f"Top Ham word            : {ham_words[0][0]} ({ham_words[0][1]} times)")
