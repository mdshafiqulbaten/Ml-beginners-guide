# 05_train_final.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Step 5: train the final model, open the test set ONCE, save everything the API
 needs."""
 import json
 import joblib
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import average_precision_score, roc_auc_score
 from sklearn.pipeline import Pipeline
 import pandas as pd
 from common import load_split, preprocessor

 # from step 4; re-run that script after changing the cost assumptions
 THRESHOLD = 0.35

 X_train, X_val, X_test, y_train, y_val, y_test = load_split()
 pipe = Pipeline([("prep", preprocessor()),
                  # the simple model won in step 3
                  ("model", LogisticRegression(max_iter=2000))])
 # Final model learns from train + validation (more data), test stays untouched until now
 pipe.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

 p = pipe.predict_proba(X_test)[:, 1]
 pred = p >= THRESHOLD
 tp = int((pred & (y_test == 1)).sum()); fp = int((pred & (y_test == 0)).sum()); fn =
 int((~pred & (y_test == 1)).sum())
 metrics = {
     "test_auc": round(float(roc_auc_score(y_test, p)), 3),
     "test_average_precision": round(float(average_precision_score(y_test, p)), 3),
     "threshold": THRESHOLD,
     "precision": round(tp / (tp + fp), 3),
     "recall": round(tp / (tp + fn), 3),
 }
 print(json.dumps(metrics, indent=2))
 joblib.dump({"pipeline": pipe, "threshold": THRESHOLD, "metrics": metrics},
 "churn_model.joblib")
 print("saved churn_model.joblib")
