# bert_features.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

internet access to download BERT). Syntax-checked only.
 """Turn text into vectors with BERT and compare meanings with cosine similarity."""
 import torch
 from transformers import AutoModel, AutoTokenizer

 name = "bert-base-uncased"
 tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()

 def embed(sentence):
     with torch.no_grad():
         out = model(**tok(sentence, return_tensors="pt"))
     return out.last_hidden_state.mean(dim=1).squeeze()            # average the token vectors

 a, b, c = embed("The cat sat on the mat."), embed("A kitten rested on the rug."),
 embed("Interest rates rose again.")
 cos = torch.nn.functional.cosine_similarity
 print("cat vs kitten :", round(cos(a, b, dim=0).item(), 3))
 print("cat vs finance:", round(cos(a, c, dim=0).item(), 3))

Visualizing Attention: What Does the Transformer
Actually Look At?
One of the most illuminating things you can do with a Transformer is visualize its
attention weights. This shows you exactly which words the model focuses on when
