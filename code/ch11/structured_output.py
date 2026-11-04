# structured_output.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Getting machine-readable answers from an LLM, and defending against bad replies.

The validation function below runs offline. Plug in your own API call where marked.
"""
import json

MODEL = "your-model-name"        # check your provider's docs for the current name

def call_llm(prompt: str) -> str:
    """Replace this body with a real API call (see Chapter 10). Returns the model's
text."""
    raise NotImplementedError

def parse_reply(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences. Strip them, parse, and validate."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    data = json.loads(text)
    rating = data.get("rating")
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError(f"rating must be an integer 1-5, got {rating!r}")
    return data

def extract_review(review: str) -> dict:
    prompt = ('Reply with ONLY JSON: {"rating": 1-5, "pros": [], "cons": []}\nReview: '
+ review)
    for attempt in range(3):                      # retry a couple of times on bad output
        try:
            return parse_reply(call_llm(prompt))
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError("model never returned valid JSON")

if __name__ == "__main__":
    good = '```json\n{"rating": 4, "pros": ["fast"], "cons": ["shallow keyboard"]}\n```'
    bad = '{"rating": "four stars"}'
    print("good reply ->", parse_reply(good))
    try:
        parse_reply(bad)
    except ValueError as e:
        print("bad reply ->", e)
