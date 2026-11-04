# initialization.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Why initialization matters: watch the signal strength through a 20-layer ReLU
network."""
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(512, 100)

def signal_after_20_layers(init):
    h = x
    for _ in range(20):
        layer = nn.Linear(100, 100, bias=False)
        init(layer.weight)
        h = torch.relu(layer(h))
    return h.std().item()

print(f"too small (std 0.01)        -> activations std {signal_after_20_layers(lambda w:
nn.init.normal_(w, std=0.01)):.2e}   (vanished)")
print(f"too large (std 0.5)         -> activations std {signal_after_20_layers(lambda w:
nn.init.normal_(w, std=0.5)):.2e}   (exploded)")
print(f"He init    (kaiming_normal_, for ReLU nets)-> {signal_after_20_layers(lambda w:
nn.init.kaiming_normal_(w, nonlinearity='relu')):.2e}   (stable)")
