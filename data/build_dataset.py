
import argparse
import json
import os
import re

import numpy as np
import pandas as pd


# models we're using
FAMILIES = {
    "GPT": ["gpt4_1106_preview", "gpt-4o-2024-05-13"],
    "Claude": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
    "Llama": ["Meta-Llama-3-70B-Instruct", "llama-2-70b-chat-hf"],
    "Qwen": ["Qwen1.5-72B-Chat", "Qwen2-72B-Instruct"],
    "Mistral": ["mistral-large-2402", "Mixtral-8x7B-Instruct-v0.1"],
    "Google": ["gemini-pro", "gemma-7b-it"]
}


# check prompts for some common words
TECH_PAT = re.compile(
    r"\b(code|coding|python|java(script)?|sql|function|programming|"
    r"algorithm|regex|html|css|bash|compile|debug|api|json|latex|"
    r"excel|formula|calculate|equation|math|integral|derivative|"
    r"probability|prime number|matrix|arithmetic)\b"
    r"|(?<!\w)(c\+\+|c#)(?!\w)"
    r"|\b\d+\s*[-+*/x^]\s*\d+\b",
    re.I
)

WRITE_PAT = re.compile(
    r"\b(write|poem|story|essay|email|letter|tweet|slogan|lyrics|"
    r"song|haiku|rewrite|paraphrase|compose|draft|blog|caption|"
    r"joke|limerick|speech|ad|advertisement|title|headline|tagline|"
    r"rhyme|creative|fiction|summari[sz]e)\b",
    re.I
)


def task_type(instruction):
    # technical takes priority if both match
    if TECH_PAT.search(instruction):
        return "technical"

    elif WRITE_PAT.search(instruction):
        return "writing"

    else:
        return "qa"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpaca_dir", default="alpaca_eval/results")
    parser.add_argument("--out", default="data/dataset.csv")
    parser.add_argument("--seed", type=int, default=448)

    args = parser.parse_args()

    rows = []

    # read all the model responses
    for family in FAMILIES:
        models = FAMILIES[family]

        for model in models:
            path = os.path.join(args.alpaca_dir, model, "model_outputs.json")

            with open(path, "r") as f:
                data = json.load(f)

            for example in data:
                row = {
                    "llm_family": family,
                    "llm_model": model,
                    "source_subset": example["dataset"],
                    "llm_input": example["instruction"],
                    "llm_output": example["output"]
                }

                rows.append(row)

    df = pd.DataFrame(rows)
    original_count = len(df)

    # get rid of empty answers
    df = df[df["llm_output"].str.strip().str.len() > 0]

    # only keep prompts that every model answered
    model_count = 0
    for family in FAMILIES:
        model_count += len(FAMILIES[family])

    counts = df.groupby("llm_input")["llm_model"].nunique()

    good_prompts = counts[counts == model_count].index
    df = df[df["llm_input"].isin(good_prompts)].copy()

    print("rows kept:", len(df), "out of", original_count)
    print("number of prompts:", df["llm_input"].nunique())

    # split by prompt so the same question isn't in train and test
    prompts = sorted(df["llm_input"].unique())

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(prompts))

    train_size = int(len(prompts) * 0.70)
    val_size = int(len(prompts) * 0.15)

    split_of = {}

    for i, idx in enumerate(order):
        prompt = prompts[idx]

        if i < train_size:
            split_of[prompt] = "train"

        elif i < train_size + val_size:
            split_of[prompt] = "val"

        else:
            split_of[prompt] = "test"

    # give each prompt an id
    prompt_ids = {}

    for i, prompt in enumerate(prompts):
        prompt_ids[prompt] = i

    df["prompt_id"] = df["llm_input"].map(prompt_ids)
    df["split"] = df["llm_input"].map(split_of)
    df["task_type"] = df["llm_input"].map(task_type)

    columns = [
        "prompt_id", "split", "task_type", "source_subset",
        "llm_family", "llm_model", "llm_input", "llm_output"
    ]

    df = df[columns]

    df.to_csv(args.out, index=False)

    print("saved dataset to", args.out)
    print(df.groupby(["split", "llm_family"]).size().unstack())

    print("task types:")
    print(df.drop_duplicates("prompt_id")["task_type"].value_counts())


if __name__ == "__main__":
    main()
