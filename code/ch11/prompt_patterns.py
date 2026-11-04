# prompt_patterns.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Prompt patterns as plain strings, plus a check of the arithmetic in the examples."""

zero_shot = "Translate to French: Good morning, how are you?"

few_shot = """Classify the sentiment.
Review: "Loved it" -> positive
Review: "Waste of money" -> negative
Review: "Fast shipping, works perfectly" ->"""

chain_of_thought = """Q: A factory makes 240 widgets per hour, works 8 hours a day for 5
days,
and 15% of widgets are defective. How many good widgets are made?
Let's think step by step."""

# The model should reason like this. We verify the numbers ourselves:
total = 240 * 8 * 5
defective = total * 0.15
print("Total widgets:     ", total)
print("Defective (15%):   ", defective)
print("Good widgets:      ", total - defective)

structured = ('Extract the product info from the review. Reply with ONLY valid JSON like ' '{"rating": 1-5, "pros": [...], "cons": [...]}')
for name, p in [("zero-shot", zero_shot), ("few-shot", few_shot), ("chain-of-thought", chain_of_thought), ("structured", structured)]:
    print(f"\n[{name}] {len(p.split())} words")
