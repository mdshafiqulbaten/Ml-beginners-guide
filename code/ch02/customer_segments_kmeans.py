# customer_segments_kmeans.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Unsupervised learning: discover customer groups nobody labeled for us."""
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")
features = ["age", "annual_spend", "purchase_frequency", "avg_order_value"]

# K-Means measures distances, so put every feature on the same scale first
X = StandardScaler().fit_transform(df[features])

# The "elbow" method: how much tighter do clusters get as k grows?
for k in range(1, 8):
    inertia = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
    print(f"k={k}: inertia = {inertia:,.0f}")

# The curve bends sharply at k=4, so use 4 clusters
df["segment"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
print()
print(df.groupby("segment")
[features].mean().round(0).assign(customers=df.segment.value_counts().sort_index()))
