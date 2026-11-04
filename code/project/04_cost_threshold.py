# 04_cost_threshold.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

code
"""Step 4: pick the decision threshold from business costs, not from habit (0.5)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from common import load_split, preprocessor

# Business assumptions: change these and watch the best threshold move.
COST_MISSED_CHURNER = 300      # lost customer value
# cost of a discount offered to someone who would have stayed anyway
COST_RETENTION_OFFER = 40
SAVE_RATE = 0.5                # fraction of true churners who accept the offer and stay

X_train, X_val, X_test, y_train, y_val, y_test = load_split()
model = Pipeline([("prep", preprocessor()), ("model",
LogisticRegression(max_iter=2000))]).fit(X_train, y_train)
p = model.predict_proba(X_val)[:, 1]
y = y_val.to_numpy()

def total_cost(threshold):
    offer = p >= threshold
    saved = offer & (y == 1)
    # every offer costs money
    cost = offer.sum() * COST_RETENTION_OFFER
    # every churner would cost us...
    cost += (y == 1).sum() * COST_MISSED_CHURNER
    # ...except those we keep
    cost -= saved.sum() * SAVE_RATE * COST_MISSED_CHURNER
    return cost

thresholds = np.arange(0.05, 0.95, 0.05)
costs = [total_cost(t) for t in thresholds]
best = thresholds[int(np.argmin(costs))]
print("threshold   total cost")
for t, c in zip(thresholds[::2], costs[::2]):
    print(f"   {t:.2f}     {c:10,.0f}")
print(f"\nDo nothing (nobody gets an offer): {total_cost(1.01):,.0f}")
print(f"Best threshold on validation data: {best:.2f} -> cost {min(costs):,.0f}")
