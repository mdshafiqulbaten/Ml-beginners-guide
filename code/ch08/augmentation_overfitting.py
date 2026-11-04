# augmentation_overfitting.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Overfitting and its cures, measured: small data, then dropout/weight decay, then
 augmentation."""
 import json
 import torch
 import torch.nn as nn
 import sys; sys.path.insert(0, "..")
 from common_digits import get_digits

 Xtr, Xte, ytr, yte = get_digits()
 Xtr, ytr = Xtr[:150], ytr[:150]                                  # deliberately tiny training
 set

 def shift(x):
     """Augmentation: move each image by up to 1 pixel in a random direction."""
     dx, dy = torch.randint(-1, 2, (2,)).tolist()
     return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

 def run(name, dropout=0.0, wd=0.0, augment=False, epochs=150, seed=0):
     torch.manual_seed(seed)
     model = nn.Sequential(nn.Flatten(), nn.Linear(64, 256), nn.ReLU(),
 nn.Dropout(dropout),
                           nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
 nn.Linear(256, 10))
     opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=wd)
     loss_fn = nn.CrossEntropyLoss()
     for _ in range(epochs):
         model.train()
         perm = torch.randperm(len(Xtr))
         for i in range(0, len(Xtr), 32):
             idx = perm[i:i + 32]
             xb = shift(Xtr[idx]) if augment else Xtr[idx]

            opt.zero_grad(); loss_fn(model(xb), ytr[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
        te = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"{name:28s} train acc {tr:.3f}   test acc {te:.3f}    gap {tr - te:+.3f}")
    return {"name": name, "train": tr, "test": te}

results = [
    run("plain network"),
    run("+ dropout 0.3", dropout=0.3),
    run("+ dropout + weight decay", dropout=0.3, wd=1e-3),
    run("+ dropout + wd + augmentation", dropout=0.3, wd=1e-3, augment=True),
]
json.dump(results, open("../outputs/ch08_regularization.json", "w"))
