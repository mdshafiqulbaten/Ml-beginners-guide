# explain_model.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

"""Which inputs does the model actually rely on? Permutation importance on the churn
model."""
import sys
sys.path.insert(0, "../project")
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from common import load_split, preprocessor

X_train, X_val, X_test, y_train, y_val, y_test = load_split()
model = Pipeline([("prep", preprocessor()), ("model",
LogisticRegression(max_iter=2000))]).fit(X_train, y_train)

# Shuffle one column at a time on VALIDATION data. The bigger the drop in AUC, the more
the model needs it.
result = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10,
random_state=0)
order = result.importances_mean.argsort()[::-1]
print("feature             AUC lost when shuffled")
for i in order:
    print(f"{X_val.columns[i]:20s}{result.importances_mean[i]:.3f} +/-
{result.importances_std[i]:.3f}")
