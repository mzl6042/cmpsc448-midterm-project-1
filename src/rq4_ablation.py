"""
RQ4 part 2: remove signals from the text and retrain the output-only CNN.
  none          : original output (= RQ1 setting)
  no_format     : markdown symbols, list markers and line breaks removed
  lower_nopunct : lowercase words only (no punctuation, no newlines, no case)
  shuffle       : words randomly shuffled (bag of words, no order)
  first50       : only the first 50 tokens of the answer
"""
import argparse
import json
import os

import pandas as pd

from train import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ablations", default="no_format,lower_nopunct,shuffle,first50")
    args = ap.parse_args()
    df = pd.read_csv("data/dataset.csv", keep_default_na=False)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    os.makedirs("results/rq4", exist_ok=True)
    path = f"results/rq4/ablation_{args.model}.json"
    out = json.load(open(path)) if os.path.exists(path) else []
    for ab in args.ablations.split(","):
        for seed in range(args.seeds):
            if os.path.exists(path):          # resume: skip runs already saved
                out = json.load(open(path))
            if any(o["ablation"] == ab and o["seed"] == seed for o in out):
                continue
            r = run_experiment(tr, va, {"test": te}, args.model, "output", seed,
                               ablation=ab, verbose=False)
            t = r["tests"]["test"]
            out.append({"ablation": ab, "seed": seed, "acc": t["acc"], "macro_f1": t["macro_f1"],
                        "per_class_f1": t["per_class_f1"]})
            print(ab, seed, round(t["acc"], 4), round(t["macro_f1"], 4), flush=True)
            json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
