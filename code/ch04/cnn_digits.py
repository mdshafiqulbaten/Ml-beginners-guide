# cnn_digits.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """A small convolutional network (CNN) on 8x8 digit images, trained on CPU in
 seconds."""
 import json

import torch
import torch.nn as nn
import sys; sys.path.insert(0, "..")
from common_digits import get_digits

torch.manual_seed(0)
Xtr, Xte, ytr, yte = get_digits()

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 8 filters, each 3x3, scan the image
    nn.ReLU(),
    nn.MaxPool2d(2),                             # 8x8 -> 4x4: keep the strongest
response
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                             # 4x4 -> 2x2
    nn.Flatten(),
    nn.Linear(16 * 2 * 2, 10),                   # 10 digit classes
)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters()))

opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(1, 31):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss_fn(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch:2d} test accuracy {acc:.3f}")

# Save the first-layer filters so they can be drawn (what the network learned to look
for)
json.dump(model[0].weight.detach().squeeze(1).tolist(),
open("../outputs/ch04_filters.json", "w"))
