# tiny_bpe.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """Subword tokenization in 30 lines: how Byte-Pair Encoding builds its vocabulary."""
 from collections import Counter

 corpus = "low low low low low lower lower newest newest newest newest newest newest

widest widest widest"
# Start with every word split into characters, plus an end-of-word marker
words = Counter(tuple(w) + ("_",) for w in corpus.split())

def count_pairs(words):
    pairs = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge(words, pair):
    new_words = Counter()
    for word, freq in words.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1]); i += 2
            else:
                out.append(word[i]); i += 1
        new_words[tuple(out)] += freq
    return new_words

for step in range(1, 9):
    pair, freq = count_pairs(words).most_common(1)[0]
    words = merge(words, pair)
    print(f"merge {step}: {pair[0]!r} + {pair[1]!r} (seen {freq} times)")

print("\nFinal splits:")
for word in words:
    print(" ", " | ".join(word))
