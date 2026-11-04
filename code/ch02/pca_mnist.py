# pca_mnist.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

"""PCA: squeeze 784 pixel values per image down to a few numbers, and see what
survives."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys; sys.path.insert(0, "..")
from common_mnist import get_mnist

Xtr, Xte, ytr, yte = get_mnist()
pca = PCA(n_components=100).fit(Xtr)                         # learned from training
images only
cum = np.cumsum(pca.explained_variance_ratio_)
for k in (2, 10, 30, 50, 100):
    print(f"{k:3d} components keep {cum[k - 1]:.0%} of the variation in the images")

for k in (2, 10, 30, 784):
    if k == 784:
        a, b = Xtr, Xte
    else:
        p = PCA(n_components=k).fit(Xtr); a, b = p.transform(Xtr), p.transform(Xte)
    acc = LogisticRegression(max_iter=3000).fit(a, ytr).score(b, yte)
    print(f"classifier on {k:3d} numbers per image: accuracy {acc:.3f}")
two = PCA(n_components=2).fit(Xtr)
np.savez("../outputs/ch02_pca.npz", cum=cum, xy=two.transform(Xte), y=yte)
