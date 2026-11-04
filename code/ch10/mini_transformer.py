# mini_transformer.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """A tiny Transformer you can read in one sitting. It learns to REVERSE a sequence of
 digits.
 To succeed, position i must attend to position (n-1-i), and we can look at the attention
 it learned."""
 import numpy as np
 import torch
 import torch.nn as nn

 torch.manual_seed(0)
 N, V, D = 8, 10, 64                                          # sequence length,
 vocabulary (digits 0-9), model width

 class Block(nn.Module):
     def __init__(self):
         super().__init__()
         self.attn = nn.MultiheadAttention(D, num_heads=1, batch_first=True)
         self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
         self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)

      def forward(self, x):
          a, w = self.attn(x, x, x, need_weights=True)        # self-attention: every
 position looks at every position
          x = self.n1(x + a)                                  # skip connection + layer
 norm
          return self.n2(x + self.ff(x)), w

 class Tiny(nn.Module):
     def __init__(self):
          super().__init__()
          self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(N, D)   # token meaning +
 position
          self.block, self.out = Block(), nn.Linear(D, V)

      def forward(self, ids):
          x = self.tok(ids) + self.pos(torch.arange(N))
          x, w = self.block(x)
          return self.out(x), w

 model = Tiny()
 opt = torch.optim.Adam(model.parameters(), lr=0.003)
 loss_fn = nn.CrossEntropyLoss()
 for step in range(1, 1501):
     ids = torch.randint(0, V, (128, N))
     loss = loss_fn(model(ids)[0].reshape(-1, V), ids.flip(1).reshape(-1))       # target:

the same digits, reversed
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        print(f"step {step:4d} loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test = torch.randint(0, V, (2000, N))
    logits, w = model(test)
    acc = (logits.argmax(-1) == test.flip(1)).float().mean().item()
    print(f"\nTest accuracy on 2,000 new random sequences: {acc:.1%}")
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    print("input :", ids[0].tolist(), "\noutput:", model(ids)[0].argmax(-1)[0].tolist())
    attn = model(test[:500])[1].mean(0).numpy()               # average attention over
many sequences
print("Where does output position 0 look? (attention weights per input position)\n",
np.round(attn[0], 2))
np.save("../outputs/ch10_attention.npy", attn)
