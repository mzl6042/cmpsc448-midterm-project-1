"""
RQ2 (iii): when can INPUT-ONLY classification work?

In our main dataset every prompt is answered by every family, so the prompt carries
zero information about the label (input-only = chance by construction).
Real scraped datasets are not like this: e.g. one team collects coding prompts from
GPT, another collects chit-chat from Llama. We simulate that "collection bias":
each prompt keeps ONE response, and the family is tied to the prompt's source
subset with probability p_bias (otherwise a random other family).
p_bias = 1/6 means no bias; we sweep it and train the input-only CNN each time.
"""
import json
import os

import numpy as np
import pandas as pd

from train import FAMILIES, run_experiment

SOURCE_TO_FAM = {"selfinstruct": "GPT", "oasst": "Claude", "koala": "Llama",
                 "helpful_base": "Qwen", "vicuna": "Mistral"}


def biased_subset(df, p_bias, seed):
    rng = np.random.RandomState(seed)
    keep = []
    for pid, g in df.groupby("prompt_id"):
        pref = SOURCE_TO_FAM[g.source_subset.iloc[0]]
        if rng.rand() < p_bias:
            fam = pref
        else:
            fam = rng.choice([f for f in FAMILIES if f != pref])
        keep.append(g[g.llm_family == fam].sample(1, random_state=rng).index[0])
    return df.loc[keep]


def main():
    df = pd.read_csv("data/dataset.csv", keep_default_na=False)
    os.makedirs("results/rq2_bias", exist_ok=True)
    out = []
    for p in [1 / 6, 0.4, 0.6, 0.8]:
        for seed in range(3):
            sub = biased_subset(df, p, seed)
            tr, va, te = (sub[sub.split == s] for s in ("train", "val", "test"))
            r = run_experiment(tr, va, {"test": te}, "cnn", "input", seed, epochs=20,
                               patience=4, batch_size=32, verbose=False)
            t = r["tests"]["test"]
            maj = te.llm_family.value_counts(normalize=True).iloc[0]
            out.append({"p_bias": p, "seed": seed, "acc": t["acc"], "macro_f1": t["macro_f1"],
                        "majority_baseline": float(maj), "n_test": len(te)})
            print(out[-1], flush=True)
    json.dump(out, open("results/rq2_bias/input_only_bias.json", "w"), indent=1)


if __name__ == "__main__":
    main()
