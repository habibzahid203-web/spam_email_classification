"""
Email Spam Classification - Model Training (Logistic Regression)
Arch Technologies Internship - Month 1 - Task 1

Ye script TF-IDF features (X_tfidf.pkl, y_labels.pkl) load karke
Logistic Regression model train karta hai aur complete evaluation deta hai.

Output:
  - spam_model.pkl              (trained model, baad mein predictions ke liye)
  - confusion_matrix.png        (visual evaluation)
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# STEP 1: TF-IDF features aur labels load karo
# ============================================================
with open("X_tfidf.pkl", "rb") as f:
    X = pickle.load(f)

with open("y_labels.pkl", "rb") as f:
    y = pickle.load(f)

print("=" * 60)
print("STEP 1: DATA LOADED")
print("=" * 60)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()

# ============================================================
# STEP 2: Train/Test Split
# ============================================================
# test_size=0.2 -> 80% training, 20% testing
# stratify=y -> spam/ham ratio train aur test dono mein same rahega (imbalance ke liye zaroori)
# random_state=42 -> har baar same split (reproducibility ke liye)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training set: {X_train.shape[0]} messages")
print(f"Testing set : {X_test.shape[0]} messages")
print()

# ============================================================
# STEP 3: Model Train karo
# ============================================================
# class_weight='balanced' -> imbalanced data (87% ham, 12% spam) ko
# properly handle karega, minority class (spam) ko zyada weight dega
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("=" * 60)
print("STEP 3: MODEL TRAINED")
print("=" * 60)
print("Logistic Regression training complete.")
print()

# ============================================================
# STEP 4: Predictions aur Evaluation
# ============================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 60)
print("STEP 4: EVALUATION METRICS")
print("=" * 60)
print(f"Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
print()

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print()

# ============================================================
# STEP 5: Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, y_pred)
print("=" * 60)
print("STEP 5: CONFUSION MATRIX")
print("=" * 60)
print(cm)
print()
print(f"True Negatives (Ham correctly identified)  : {cm[0][0]}")
print(f"False Positives (Ham wrongly marked Spam)  : {cm[0][1]}")
print(f"False Negatives (Spam wrongly marked Ham)  : {cm[1][0]}")
print(f"True Positives (Spam correctly identified) : {cm[1][1]}")
print()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")
plt.show()

# ============================================================
# STEP 6: Trained model save karo
# ============================================================
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)
print("spam_model.pkl -> trained Logistic Regression model")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Algorithm        : Logistic Regression (class_weight='balanced')")
print(f"Train/Test Split : 80% / 20%")
print(f"Accuracy         : {accuracy*100:.2f}%")
print(f"Precision        : {precision*100:.2f}%")
print(f"Recall           : {recall*100:.2f}%")
print(f"F1-Score         : {f1*100:.2f}%")
