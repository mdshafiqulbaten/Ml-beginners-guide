# data_leakage_demo.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

"""Data leakage, measured. The data below is PURE NOISE, so no honest model can beat
50%."""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))                 # 100 people, 5,000 random features
y = rng.integers(0, 2, size=100)                 # random labels: there is nothing to learn

# WRONG: choose the 20 'best' features using ALL the data, then cross-validate
X_leaky = SelectKBest(f_classif, k=20).fit_transform(X, y)
leaky = cross_val_score(LogisticRegression(), X_leaky, y, cv=5).mean()

# RIGHT: do the selection inside the pipeline, so each fold picks features from its own
training part only
honest_model = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
honest = cross_val_score(honest_model, X, y, cv=5).mean()

print(f"Leaky accuracy (selection before the split): {leaky:.2f}   <- looks like a
discovery")
print(f"Honest accuracy (selection inside the folds): {honest:.2f}   <- the truth: a
coin flip")
