# full_evaluation.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""A complete model report: more than just accuracy."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                            confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def full_evaluation(model, X_test, y_test, class_names, threshold=0.5):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    print(classification_report(y_test, pred, target_names=class_names))
    print(f"AUC-ROC           {roc_auc_score(y_test, prob):.4f}")
    print(f"Average precision {average_precision_score(y_test, prob):.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}")

    print("\nThreshold analysis (how the trade-off moves):")

for t in [0.2, 0.35, 0.5, 0.65, 0.8]:
        p = (prob >= t).astype(int)
        tp_t = int(((p == 1) & (y_test == 1)).sum()); fp_t = int(((p == 1) & (y_test ==
0)).sum()); fn_t = int(((p == 0) & (y_test == 1)).sum())
        precision = tp_t / (tp_t + fp_t) if tp_t + fp_t else 0.0
        recall = tp_t / (tp_t + fn_t) if tp_t + fn_t else 0.0
        print(f" threshold {t:.2f}: precision {precision:.3f} recall {recall:.3f}")

X, y = load_breast_cancer(return_X_y=True)
# make 1 = malignant, the case we must not miss
y = 1 - y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
random_state=1)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_train,
y_train)
full_evaluation(model, X_test, y_test, ["benign", "malignant"])
