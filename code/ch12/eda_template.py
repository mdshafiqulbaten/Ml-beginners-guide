# eda_template.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""A reusable first look at any tabular dataset (shown on the breast cancer data)."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "benign"})          # 1 = benign, 0 =
malignant

# 1. First look
print("shape:", df.shape)
print("missing values:", int(df.isnull().sum().sum()))

# 2. Target balance. Is one class rare?
print("\nclass balance:\n", df["benign"].value_counts(normalize=True).round(3))

# 3. Which features relate most to the target?
corr = df.corr()["benign"].drop("benign").sort_values()
print("\nmost negatively correlated:\n", corr.head(3).round(2))
print("\nmost positively correlated:\n", corr.tail(3).round(2))

# 4. Pictures: distributions and a correlation heatmap
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, col in zip(axes, ["mean radius", "mean texture", "worst perimeter"]):
    sns.histplot(data=df, x=col, hue="benign", ax=ax, bins=25)
fig.tight_layout(); fig.savefig("eda_distributions.png", dpi=120)
