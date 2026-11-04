# numpy_basics.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Vectors, matrices and the dot product: the language of ML data."""
import numpy as np

# A vector: one example described by 3 numbers [temperature, humidity, wind]
x = np.array([35, 0.78, 120])

# A matrix: 5 examples (rows) x 3 features (columns)
X = np.array([
    [35, 0.78, 120],
    [28, 0.60, 80],
    [40, 0.90, 200],
    [22, 0.45, 60],
    [31, 0.70, 100],
])
print("Shape of X:", X.shape)            # (5, 3)

# Apply the same weights to all 5 examples at once with one matrix product
weights = np.array([0.5, 0.3, -0.2])
predictions = X @ weights                # @ means matrix multiplication
print("One score per example:", predictions.round(2))

# Dot product: multiply pairwise, then add up. It measures similarity.
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
print("Dot product:", np.dot(a, b))      # 1*4 + 2*5 + 3*6 = 32
