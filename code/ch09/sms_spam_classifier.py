# sms_spam_classifier.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

 """A real text classifier: spam detection, judged the way a business would judge it."""
 import numpy as np
 import pandas as pd
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import classification_report, confusion_matrix
 from sklearn.model_selection import train_test_split
 from sklearn.pipeline import make_pipeline

 sms = pd.read_csv("../data/real/sms_spam.tsv", sep="\t", header=None, names=["label",
 "text"], quoting=3)
 y = (sms.label == "spam").astype(int)
 X_train, X_test, y_train, y_test = train_test_split(sms.text, y, test_size=0.2,
 stratify=y, random_state=42)

 # TF-IDF turns each message into numbers: words that are rare overall but frequent here
 get more weight
 model = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
 sublinear_tf=True),
                       LogisticRegression(C=10, class_weight="balanced", max_iter=2000))
 model.fit(X_train, y_train)

 p = model.predict_proba(X_test)[:, 1]
 for threshold in (0.5, 0.8):
     pred = (p >= threshold).astype(int)
     tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
     print(f"threshold {threshold}: caught {tp} of {tp + fn} spam, but blocked {fp}
 genuine messages")
 print()
 print(classification_report(y_test, (p >= 0.5).astype(int), target_names=["ham",
 "spam"]))

 # What did the model learn? The words with the largest weights
 vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
 words = np.array(vec.get_feature_names_out())
 top = np.argsort(clf.coef_[0])
 print("words most typical of SPAM:", list(words[top[-8:]][::-1]))
 print("words most typical of HAM: ", list(words[top[:8]]))
 np.savez("../outputs/ch09_sms.npz", spam_words=words[top[-10:]][::-1],
 spam_w=clf.coef_[0][top[-10:]][::-1],
          ham_words=words[top[:10]], ham_w=clf.coef_[0][top[:10]], p=p,
 y=y_test.to_numpy())
