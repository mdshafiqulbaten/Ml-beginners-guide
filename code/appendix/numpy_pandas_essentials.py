# numpy_pandas_essentials.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Appendix A: the NumPy and pandas you actually need, with every result printed."""
import numpy as np
import pandas as pd

a = np.array([10, 20, 30, 40, 50])
print("first, last, slice:", a[0], a[-1], a[1:4])
print("boolean filter a[a > 25]:", a[a > 25])

x, w = np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])
print("element-wise x + w:", x + w, " dot product:", x @ w)

# Broadcasting: a (3,1) column and a (1,4) row combine into a (3,4) table
col, row = np.array([[0], [10], [20]]), np.array([[1, 2, 3, 4]])
print("broadcast shape:", (col + row).shape, "\n", col + row)

df = pd.DataFrame({"city": ["Dhaka", "Oslo", "Lima", "Dhaka", "Oslo"],
                   "sales": [120, 80, np.nan, 150, 95],
                   "units": [12, 8, 5, 15, 9]})
print("\nmissing per column:\n", df.isnull().sum().to_string())
df["sales"] = df["sales"].fillna(df["sales"].median())
print("\nsales by city:\n", df.groupby("city")["sales"].agg(["mean",

"sum"]).to_string())
df["price_per_unit"] = df["sales"] / df["units"]
print("\nhigh-price rows:\n", df[df["price_per_unit"] > 10].to_string())
