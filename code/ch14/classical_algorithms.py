# classical_algorithms.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """The classical algorithms side by side, on the same data, with the same honest
 test."""
 from sklearn.datasets import load_breast_cancer
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedKFold, cross_val_score
 from sklearn.naive_bayes import GaussianNB
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.svm import SVC
 from sklearn.tree import DecisionTreeClassifier

 X, y = load_breast_cancer(return_X_y=True)
 cv = StratifiedKFold(5, shuffle=True, random_state=0)

 models = {
     "Logistic regression": make_pipeline(StandardScaler(),
 LogisticRegression(max_iter=3000)),
     "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
     "k-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
     "Support Vector Machine": make_pipeline(StandardScaler(), SVC()),
     "Naive Bayes": GaussianNB(),
 }
 for name, model in models.items():
     scores = cross_val_score(model, X, y, cv=cv)
     print(f"{name:25s} accuracy {scores.mean():.3f} +/- {scores.std():.3f}")
 # Note: KNN and SVM need scaled features (they measure distances); trees and Naive Bayes
 do not.
