# debugging_checks.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Three checks that catch most training bugs, before you waste hours of compute."""
import torch
import torch.nn as nn
import sys; sys.path.insert(0, "..")
from common_digits import get_digits

torch.manual_seed(0)
Xtr, Xte, ytr, yte = get_digits()
Xtr = Xtr.flatten(1)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Check 1: the starting loss should be about ln(10) = 2.303 for 10 balanced classes
start = loss_fn(model(Xtr), ytr).item()
print(f"1. initial loss {start:.3f}   (expected about {torch.log(torch.tensor(10.)).item():.3f})")

# Check 2: the model must be able to memorize ONE tiny batch (loss -> ~0)
xb, yb = Xtr[:16], ytr[:16]
for _ in range(200):
    opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
print(f"2. loss after overfitting 16 examples: {loss.item():.4f}   (should be near 0)")

# Check 3: gradients should exist and be neither zero nor huge in every layer
opt.zero_grad(); loss_fn(model(Xtr), ytr).backward()
print("3. gradient size per parameter tensor:")
for name, p in model.named_parameters():
    print(f"   {name:10s} mean |grad| = {p.grad.abs().mean().item():.5f}")
