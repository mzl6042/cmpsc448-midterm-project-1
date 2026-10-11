"""
Build the (LLM_name, LLM_input, LLM_output) dataset for the project.

Source: AlpacaEval (tatsu-lab/alpaca_eval on GitHub), which publishes the raw
outputs of many LLMs on the SAME 805 evaluation instructions in
results/<model>/model_outputs.json. We only use official vendor models
(no community fine-tunes) and group them into 6 families, 2 models each,
so every family has the same number of examples.

Usage:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/tatsu-lab/alpaca_eval.git
    cd alpaca_eval && git sparse-checkout set results --no-cone && cd ..
    python data/build_dataset.py --alpaca_dir alpaca_eval/results
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# family -> AlpacaEval result folder names
FAMILIES = {
    "GPT":     ["gpt4_1106_preview", "gpt-4o-2024-05-13"],
    "Claude":  ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
    "Llama":   ["Meta-Llama-3-70B-Instruct", "llama-2-70b-chat-hf"],
    "Qwen":    ["Qwen1.5-72B-Chat", "Qwen2-72B-Instruct"],
    "Mistral": ["mistral-large-2402", "Mixtral-8x7B-Instruct-v0.1"],
    "Google":  ["gemini-pro", "gemma-7b-it"],
}

# --- simple keyword rules to tag each prompt with a task type (used for RQ3) ---
TECH_PAT = re.compile(
    r"\b(code|coding|python|java(script)?|c\+\+|c#|sql|function|programming|algorithm|regex|"
    r"html|css|bash|compile|debug|api|json|latex|excel|formula|calculate|equation|"
    r"math|integral|derivative|probability|prime number|matrix|arithmetic|"
    r"\d+\s*[\+\-\*/x\^]\s*\d+)\b",
    re.I)
WRITE_PAT = re.compile(
    r"\b(write|poem|story|essay|email|letter|tweet|slogan|lyrics|song|haiku|rewrite|"
    r"paraphrase|compose|draft|blog|caption|joke|limerick|speech|ad\b|advertisement|"
    r"title|headline|tagline|rhyme|creative|fiction|summari[sz]e)\b",
    re.I)


def task_type(instruction):
    if TECH_PAT.search(instruction):
        return "technical"   # coding / math / data
    if WRITE_PAT.search(instruction):
        return "writing"     # creative + professional writing
    return "qa"              # open questions, advice, explanations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca_dir", default="alpaca_eval/results")
    ap.add_argument("--out", default="data/dataset.csv")
    ap.add_argument("--seed", type=int, default=448)
    args = ap.parse_args()

    rows = []
    for fam, models in FAMILIES.items():
        for m in models:
            with open(os.path.join(args.alpaca_dir, m, "model_outputs.json")) as f:
                data = json.load(f)
            for ex in data:
                rows.append({
                    "llm_family": fam,
                    "llm_model": m,
                    "source_subset": ex["dataset"],
                    "llm_input": ex["instruction"],
                    "llm_output": ex["output"],
                })
    df = pd.DataFrame(rows)
    n0 = len(df)

    # cleaning: drop empty outputs, then keep only prompts answered by ALL 12 models
    df = df[df["llm_output"].str.strip().str.len() > 0]
    counts = df.groupby("llm_input")["llm_model"].nunique()
    full = counts[counts == sum(len(v) for v in FAMILIES.values())].index
    df = df[df["llm_input"].isin(full)].copy()
    print(f"kept {len(df)}/{n0} rows, {df['llm_input'].nunique()} prompts")

    # prompt ids + split by PROMPT (all 12 answers of a prompt go to the same split)
    prompts = sorted(df["llm_input"].unique())
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(prompts))
    n_tr, n_va = int(0.70 * len(prompts)), int(0.15 * len(prompts))
    split_of = {}
    for rank, idx in enumerate(perm):
        split_of[prompts[idx]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    pid = {p: i for i, p in enumerate(prompts)}
    df["prompt_id"] = df["llm_input"].map(pid)
    df["split"] = df["llm_input"].map(split_of)
    df["task_type"] = df["llm_input"].map(task_type)

    df = df[["prompt_id", "split", "task_type", "source_subset", "llm_family",
             "llm_model", "llm_input", "llm_output"]]
    df.to_csv(args.out, index=False)
    print(df.groupby(["split", "llm_family"]).size().unstack())
    print(df.drop_duplicates("prompt_id")["task_type"].value_counts())


if __name__ == "__main__":
    main()
