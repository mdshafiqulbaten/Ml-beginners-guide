# gradient_descent_numpy.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Gradient descent from scratch: fit a line to real data with three learning rates."""
import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
x = X[:, 2]
# standardize the input so one learning rate suits both weights
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()                  # scale the target so numbers stay small

def train(lr, steps=200):
    w, b = 0.0, 0.0
    for step in range(steps):
        error = (w * x + b) - y
        loss = np.mean(error ** 2)
        if not np.isfinite(loss) or loss > 1e6:
            return float("inf"), w, b
        grad_w = 2 * np.mean(error * x)       # d(loss)/dw
        grad_b = 2 * np.mean(error)           # d(loss)/db
        w -= lr * grad_w                      # step downhill
        b -= lr * grad_b
    return loss, w, b

for lr in [0.001, 0.05, 0.5, 1.05]:
    loss, w, b = train(lr)
    print(f"learning rate {lr:>6}: final loss = {loss:.4f}            w = {w:.3f}")

# Compare with the exact least-squares answer

w_exact, b_exact = np.polyfit(x, y, 1)
print(f"exact solution:           w = {w_exact:.3f}         b = {b_exact:.3f}")
