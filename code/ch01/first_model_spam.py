# first_model_spam.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: passed.
# Review before running, especially if the status above is not "passed".

"""Your first real ML program: a spam filter trained on 5,574 real text messages."""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Load real messages: each line is a label ('ham' = normal, 'spam') and the text
sms = pd.read_csv("../data/real/sms_spam.tsv", sep="\t", header=None, names=["label",
"text"], quoting=3)
print(f"{len(sms)} messages, {(sms.label == 'spam').mean():.1%} are spam")

# 2. Hold back 20% of the messages that the model will never see during learning
train_text, test_text, train_y, test_y = train_test_split(
    sms.text, sms.label, test_size=0.2, stratify=sms.label, random_state=42)

# 3. Turn words into counts, then learn which words signal spam
vectorizer = CountVectorizer()
model = MultinomialNB().fit(vectorizer.fit_transform(train_text), train_y)

# 4. Test on the hidden messages
accuracy = model.score(vectorizer.transform(test_text), test_y)
print(f"Accuracy on {len(test_text)} unseen messages: {accuracy:.1%}")

# 5. Try it on messages you write yourself
for msg in ["WINNER!! You have won a 1000 pound prize. Call 09061701461 now to claim",
            "Are we still meeting for lunch at 1pm tomorrow?"]:
    print(f"{model.predict(vectorizer.transform([msg]))[0]:>5} <- {msg}")
