# dropout_demo.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Dropout behaves differently in training and evaluation mode. Forgetting model.eval()
is a classic bug."""
import torch
import torch.nn as nn

torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.ones(10)

drop.train()
print("train mode:", drop(x).tolist(), " <- about half zeroed, survivors scaled x2")
drop.eval()
print("eval mode: ", drop(x).tolist(), " <- untouched")
# Scaling by 1/(1-p) in training keeps the average activation the same in both modes.
