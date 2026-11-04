# transfer_learning.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

internet access to download pretrained weights). Syntax-checked only.
 """Transfer learning: reuse a network trained on ImageNet, retrain only the last
 layer."""
 import torch.nn as nn
 import torchvision.models as models

 # downloads weights on first use
 model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
 for p in model.parameters():
     p.requires_grad = False                                         # freeze
 everything...
 # ...then add a fresh head for 5 classes
 model.fc = nn.Linear(model.fc.in_features, 5)
 trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
 # only the new head is trained
 print("trainable parameters:", trainable)
 # Next: train as in Chapter 5 using only model.fc.parameters() in the optimizer.
 # When you have more data, unfreeze the last block and use a 10x smaller learning rate
 for it.

Data Augmentation — Multiplying Your Dataset for
Free
Data augmentation applies random transformations to training images, creating
variations that teach the model to be invariant to those changes. A model trained

with augmentation generalizes dramatically better because it has seen the same
objects in many different conditions.
