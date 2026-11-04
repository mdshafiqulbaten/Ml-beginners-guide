# stacking.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Stacking: let a small 'meta' model learn how much to trust each base model."""
 from sklearn.datasets import load_breast_cancer
 from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,
 StackingClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import StratifiedKFold, cross_val_score
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.svm import SVC

 X, y = load_breast_cancer(return_X_y=True)
 cv = StratifiedKFold(5, shuffle=True, random_state=0)

 base = [
     ("forest", RandomForestClassifier(200, random_state=0)),
     ("boost", GradientBoostingClassifier(random_state=0)),
     ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
 ]
 # StackingClassifier builds the out-of-fold predictions for us, so there is no leakage
 stack = StackingClassifier(base, final_estimator=LogisticRegression(), cv=5)

 for name, model in base + [("STACK", stack)]:
     auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
     print(f"{name:7s} AUC {auc.mean():.4f} +/- {auc.std():.4f}")
