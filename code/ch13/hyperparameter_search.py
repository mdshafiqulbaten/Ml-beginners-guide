# hyperparameter_search.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Random search: try many settings, keep the best (judged by cross-validation)."""
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
random_state=42)

search_space = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(2, 6),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.6, 0.4),             # 0.6 to 1.0
    "min_samples_leaf": randint(1, 10),
}
search = RandomizedSearchCV(GradientBoostingClassifier(random_state=0), search_space,
n_iter=20,
                             cv=5, scoring="roc_auc", n_jobs=-1, random_state=42)
search.fit(X_train, y_train)                    # the test set is NOT used here

print("Best settings:", {k: round(float(v), 3) for k, v in search.best_params_.items()})
print(f"Best cross-validated AUC: {search.best_score_:.4f}")
print(f"Test AUC (touched once):   {roc_auc_score(y_test, search.predict_proba(X_test)
[:, 1]):.4f}")
