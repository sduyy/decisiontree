"""
Compare custom C++ Decision Tree with sklearn Decision Tree
Exports results to comparison_report.txt
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

# Load data
train_df = pd.read_csv("data/train.csv", header=None, names=["label", "a", "b", "c", "d"])
test_df = pd.read_csv("data/test.csv", header=None, names=["a", "b", "c", "d"])

# Open report file
report_file = open("reports/comparison_report.txt", "w")

def log(msg=""):
    print(msg)
    report_file.write(msg + "\n")

# Header
log("=" * 70)
log("DECISION TREE COMPARISON: Custom C++ vs sklearn")
log("=" * 70)

# Prepare data
X_train = train_df[["a", "b", "c", "d"]]
y_train = train_df["label"]
X_test = test_df[["a", "b", "c", "d"]]

log(f"\nDataset Info:")
log(f"  Train samples: {len(X_train)}")
log(f"  Test samples: {len(X_test)}")
log(f"  Features: {list(X_train.columns)}")
log(f"  Classes: {sorted(y_train.unique())}")

# ============================================================================
# 1. Sklearn Decision Tree
# ============================================================================
log("\n" + "=" * 70)
log("SKLEARN DECISION TREE")
log("=" * 70)

sklearn_dt = DecisionTreeClassifier(random_state=42)
sklearn_dt.fit(X_train, y_train)

sklearn_train_pred = sklearn_dt.predict(X_train)
sklearn_test_pred = sklearn_dt.predict(X_test)

sklearn_train_acc = accuracy_score(y_train, sklearn_train_pred)
log(f"\nTraining Accuracy: {sklearn_train_acc:.4f}")
log(f"  (Correct: {np.sum(sklearn_train_pred == y_train)} / {len(y_train)})")

log(f"\nTree Properties:")
log(f"  Tree depth: {sklearn_dt.get_depth()}")
log(f"  Number of leaves: {sklearn_dt.get_n_leaves()}")
log(f"  Total nodes: {sklearn_dt.tree_.node_count}")

log(f"\nFeature Importance:")
for feat, imp in zip(X_train.columns, sklearn_dt.feature_importances_):
    log(f"  {feat}: {imp:.4f}")

sklearn_pred_df = pd.DataFrame({
    "ID": range(1, len(sklearn_test_pred) + 1),
    "Label_sklearn": sklearn_test_pred
})

log(f"\nTest Predictions (first 10):")
log(sklearn_pred_df.head(10).to_string(index=False))



# ============================================================================
# 2. C++ Decision Tree
# ============================================================================
log("\n" + "=" * 70)
log("CUSTOM C++ DECISION TREE")
log("=" * 70)

if os.path.exists("data/predict.txt"):
    with open("data/predict.txt", "r") as f:
        cpp_test_pred = [line.strip() for line in f.readlines()]
    
    log(f"\nTest Predictions (first 10):")
    for i, pred in enumerate(cpp_test_pred[:10], 1):
        log(f"  {i}: {pred}")

else:
    log("❌ ERROR: data/predict.txt not found!")
    log("   Please run your C++ decision tree first to generate predictions.")
    cpp_test_pred = None

# ============================================================================
# 3. Comparison
# ============================================================================
if cpp_test_pred is not None and len(cpp_test_pred) == len(sklearn_test_pred):
    log("\n" + "=" * 70)
    log("COMPARISON: Predictions Match Analysis")
    log("=" * 70)
    
    sklearn_pred_str = list(sklearn_test_pred)
    matches = sum(1 for s, c in zip(sklearn_pred_str, cpp_test_pred) if s == c)
    match_rate = matches / len(cpp_test_pred) * 100
    
    log(f"\nMatching predictions: {matches} / {len(cpp_test_pred)} ({match_rate:.2f}%)")
    
    differences = [(i, s, c) for i, (s, c) in enumerate(zip(sklearn_pred_str, cpp_test_pred), 1) if s != c]
    if differences:
        log(f"\nDifferences found: {len(differences)}")
        log(f"\n{'Index':<8} {'sklearn':<12} {'Your C++':<12}")
        log("-" * 32)
        for idx, sklearn, cpp in differences:
            log(f"{idx:<8} {sklearn:<12} {cpp:<12}")
    else:
        log("✓ All predictions match perfectly!")

# Close report file
report_file.close()
print("\nReport saved to: reports/comparison_report.txt")
