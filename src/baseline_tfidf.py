"""Non-neural reference point: TF-IDF (word 1-2grams + char 2-5grams) + logistic regression.
Not required by the assignment, but useful to judge whether the CNN/LSTM numbers are good."""
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/dataset.csv", keep_default_na=False)
tr, te = df[df.split.isin(["train", "val"])], df[df.split == "test"]
res = {}
for mode in ["output", "input", "both"]:
    get = {"output": lambda d: d.llm_output,
           "input": lambda d: d.llm_input,
           "both": lambda d: d.llm_input + " <sep> " + d.llm_output}[mode]
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True,
                         lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    cv = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=200000,
                         sublinear_tf=True)
    Xtr = hstack([wv.fit_transform(get(tr)), cv.fit_transform(get(tr))]).tocsr()
    Xte = hstack([wv.transform(get(te)), cv.transform(get(te))]).tocsr()
    clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr, tr.llm_family)
    p = clf.predict(Xte)
    res[mode] = {"acc": accuracy_score(te.llm_family, p),
                 "macro_f1": f1_score(te.llm_family, p, average="macro")}
    print(mode, res[mode], flush=True)
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/baseline_tfidf.json", "w"), indent=1)
