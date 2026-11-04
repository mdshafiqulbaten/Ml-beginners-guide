# feature_selection.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Keep the features that matter, drop the rest."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
random_state=42)

# Learn which features matter USING THE TRAINING DATA ONLY
selector = SelectFromModel(RandomForestClassifier(300, random_state=42),
threshold="mean")
selector.fit(X_train, y_train)
kept = X.columns[selector.get_support()]
print(f"Kept {len(kept)} of {X.shape[1]} features:")
print(list(kept))

rf = RandomForestClassifier(300, random_state=42)
all_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
sel_score = cross_val_score(rf, selector.transform(X_train), y_train, cv=5).mean()
print(f"\ncross-val accuracy, all features:       {all_score:.3f}")
print(f"cross-val accuracy, selected features: {sel_score:.3f}")
