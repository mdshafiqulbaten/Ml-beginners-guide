# representation_bias.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Bias experiment: a model that looks great overall can fail badly for a small group.

 We build a dataset with a majority group A (90%) and a minority group B (10%) in which
 the *relationship between features and outcome differs* between the groups.
 """
 import numpy as np
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import accuracy_score
 from sklearn.model_selection import train_test_split

 rng = np.random.default_rng(0)
 n = 10_000
 group = rng.choice(["A", "B"], size=n, p=[0.9, 0.1])
 x = rng.normal(size=(n, 2))
 # Group A: the outcome depends on feature 0. Group B: it depends on feature 1.
 signal = np.where(group == "A", x[:, 0], x[:, 1])
 y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)

 X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(x, y, group,
 test_size=0.3, random_state=0)
 model = LogisticRegression().fit(X_train, y_train)
 pred = model.predict(X_test)

 print(f"Overall accuracy: {accuracy_score(y_test, pred):.3f}   <- the number that goes
 in the slide deck")
 for g in ["A", "B"]:
     m = g_test == g
     print(f"Group {g} accuracy: {accuracy_score(y_test[m], pred[m]):.3f}   ({m.sum()}
 test examples)")

 # Fix attempt: give the model the group and an interaction, so it can learn both
 patterns
 def with_group(X, g):
     isB = (g == "B").astype(float)[:, None]
     return np.hstack([X, isB, X * isB])

 model2 = LogisticRegression().fit(with_group(X_train, g_train), y_train)
 pred2 = model2.predict(with_group(X_test, g_test))
 print("\nAfter letting the model see group membership and learn group-specific
 patterns:")
 for g in ["A", "B"]:
     m = g_test == g
     print(f"Group {g} accuracy: {accuracy_score(y_test[m], pred2[m]):.3f}")
 print("\nLesson: always report metrics per group. Whether using group membership is
 acceptable is a legal and ethical question, not just a technical one.")
