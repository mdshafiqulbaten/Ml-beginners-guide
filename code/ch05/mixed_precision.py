# mixed_precision.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

NVIDIA GPU). Syntax-checked only.
 """Automatic mixed precision (AMP): faster training on modern GPUs, same code shape."""
 import torch
 import torch.nn as nn

 device = "cuda"
 model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
 opt = torch.optim.Adam(model.parameters())
 # rescales the loss so tiny float16 gradients do not vanish
 scaler = torch.amp.GradScaler("cuda")
 loss_fn = nn.CrossEntropyLoss()

 x = torch.randn(128, 64, device=device)
 y = torch.randint(0, 10, (128,), device=device)
 for step in range(100):
     opt.zero_grad()
     with torch.autocast("cuda", dtype=torch.float16):            # forward pass in half
 precision
         loss = loss_fn(model(x), y)
     scaler.scale(loss).backward()
     scaler.step(opt)
     scaler.update()
 print("final loss", loss.item())

A Complete Production-Quality Training Script
Let us write a complete, well-structured training script that incorporates all best
practices: logging, checkpointing, early stopping, learning rate scheduling, and mixed
