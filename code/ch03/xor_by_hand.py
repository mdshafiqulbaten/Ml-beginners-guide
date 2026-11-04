# xor_by_hand.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""XOR: impossible for one neuron, easy for a network with one hidden layer."""
import numpy as np

step = lambda z: (z > 0).astype(int)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Hidden layer: neuron 1 computes OR, neuron 2 computes AND
W_hidden = np.array([[1, 1], [1, 1]])
b_hidden = np.array([-0.5, -1.5])
# Output neuron: "OR but not AND" (that is exactly XOR)
W_out = np.array([1, -1])
b_out = -0.5

hidden = step(inputs @ W_hidden + b_hidden)
output = step(hidden @ W_out + b_out)
for a, b, o in zip(inputs[:, 0], inputs[:, 1], output):
    print(f"{a} XOR {b} = {o}")
