# ensemble_math.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

"""Why voting works: the exact accuracy of a majority vote of independent models."""
from scipy.stats import binom

n = 100                                      # number of models
print(f"Majority vote of {n} independent classifiers:")
for p in [0.51, 0.55, 0.60, 0.65, 0.70, 0.80]:
    accuracy = 1 - binom.cdf(n // 2, n, p)     # P(more than 50 of 100 are right)
    print(f" each model {p:.0%} accurate -> ensemble {accuracy:.4%}")

# The catch: real models share training data, so their mistakes are correlated.
# Perfectly correlated models gain nothing: if all 100 copy one model, accuracy stays at
p.
