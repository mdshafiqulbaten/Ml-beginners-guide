# neuron_from_scratch.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""One artificial neuron, built with nothing but NumPy."""
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) * 0.01      # small random start
        self.bias = 0.0

        @staticmethod
        def relu(z):
            return max(0.0, z)              # pass positives, block negatives

        def forward(self, inputs):
            z = np.dot(self.weights, inputs) + self.bias    # w1*x1 + w2*x2 + ... + b
            return z, self.relu(z)

# A "weather neuron" with the same numbers used in the text
neuron = Neuron(num_inputs=3)
neuron.weights = np.array([0.5, 0.3, -0.2])
neuron.bias = -20

z, output = neuron.forward([35, 80, 10])       # temperature, humidity, wind
print(f"weighted sum z = {z:.1f}")
print(f"after ReLU     = {output:.1f}    (positive, so the neuron 'fires')")

# Why activation functions matter: stacking linear layers adds nothing
W1, W2 = np.random.randn(4, 3), np.random.randn(2, 4)
x = np.random.randn(3)
two_layers = W2 @ (W1 @ x)
one_layer = (W2 @ W1) @ x
print("Two linear layers == one linear layer:", np.allclose(two_layers, one_layer))
