# math_in_code.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Appendix B: the maths of ML as short, checkable code."""
import numpy as np

# Vectors and the dot product: a weighted sum, the core move of every neuron
features = np.array([35, 75000, 1.0])
weights = np.array([0.02, 0.00001, 0.5])
print("weighted sum:", features @ weights)

# Derivative as 'how much does the output move if I nudge the input?'
def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: x ** 2 + 3 * x
print("slope of x^2 + 3x at x=2: numerical", round(numerical_gradient(f, 2.0), 4), " exact", 2 * 2 + 3)  # Chain rule check: d/dx sigmoid(2x) = 2 * s * (1 - s)  sigmoid = lambda z: 1 / (1 + np.exp(-z)) x0 = 0.7 s = sigmoid(2 * x0) print("chain rule:", round(numerical_gradient(lambda x: sigmoid(2 * x), x0), 6), "vs", round(2 * s * (1 - s), 6))  # Probability by simulation vs by formula rng = np.random.default_rng(0) rolls = rng.integers(1, 7, size=100_000) print("P(6) simulated:", round((rolls == 6).mean(), 4), " exact:", round(1 / 6, 4)) print("P(at least one 6 in 4 rolls): exact", round(1 - (5 / 6) ** 4, 4), " simulated", round((rng.integers(1, 7, size=(100_000, 4)) == 6).any(axis=1).mean(), 4))  # Bayes: a 1%-prevalence disease, 95%-sensitive, 5% false-positive test prior, sens, fpr = 0.01, 0.95, 0.05 posterior = sens * prior / (sens * prior + fpr * (1 - prior)) print(f"P(disease | positive test) = {posterior:.3f} (not 0.95!)")
