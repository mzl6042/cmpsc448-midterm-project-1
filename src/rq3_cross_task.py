"""
RQ3: do LLM fingerprints generalize across task types?

Prompts are tagged technical / writing / qa (see data/build_dataset.py).
For each held-out task type T:
  cross  : train on train-split prompts NOT of type T  -> test on test-split prompts of type T
  in-dom : train on a random sample of train-split prompts of the SAME size,
           drawn from all types (so it does see type T)        -> same test set
Same test examples, same amount of training data, so the gap = cost of never seeing T.
Output-only CNN (the RQ1 setting), 3 seeds.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from train import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv("data/dataset.csv", keep_default_na=False)
    tr_all, va_all, te_all = (df[df.split == s] for s in ("train", "val", "test"))
    os.makedirs("results/rq3", exist_ok=True)
    out = []
    for T in ["technical", "writing", "qa"]:
        test = {T: te_all[te_all.task_type == T]}
        cross_tr = tr_all[tr_all.task_type != T]
        cross_va = va_all[va_all.task_type != T]
        n_prompts = cross_tr.prompt_id.nunique()
        for seed in range(args.seeds):
            rng = np.random.RandomState(100 + seed)
            pids = rng.choice(tr_all.prompt_id.unique(), n_prompts, replace=False)
            in_tr = tr_all[tr_all.prompt_id.isin(pids)]
            for cond, tr, va in [("cross", cross_tr, cross_va), ("in_domain", in_tr, va_all)]:
                r = run_experiment(tr, va, test, args.model, "output", seed, verbose=False)
                t = r["tests"][T]
                row = {"held_out": T, "cond": cond, "seed": seed, "n_train": len(tr),
                       "acc": t["acc"], "macro_f1": t["macro_f1"],
                       "per_class_f1": t["per_class_f1"], "n_test": t["n"]}
                out.append(row)
                print({k: v for k, v in row.items() if k != "per_class_f1"}, flush=True)
                json.dump(out, open(f"results/rq3/cross_task_{args.model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
