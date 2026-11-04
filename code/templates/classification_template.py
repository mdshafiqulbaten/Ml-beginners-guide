# classification_template.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Copy-and-adapt template: a complete, leak-free binary classification workflow."""
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. LOAD (replace with pd.read_csv("your.csv"))
data = load_breast_cancer(as_frame=True)
df, TARGET = data.frame, "target"

# 2. SPLIT FIRST, before looking at anything
X, y = df.drop(columns=TARGET), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
random_state=42)

# 3. PREPROCESS inside a pipeline so it is fit on training data only
num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale",
StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(300,
random_state=0))])

# 4. VALIDATE with cross-validation, 5. TEST once
print("CV AUC:", cross_val_score(pipe, X_train, y_train, cv=5,
scoring="roc_auc").mean().round(3))
pipe.fit(X_train, y_train)
print("Test AUC:", round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))
print(classification_report(y_test, pipe.predict(X_test)))
