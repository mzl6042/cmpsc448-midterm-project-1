
import argparse
import json
import os

import numpy as np
import pandas as pd

from train import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv("data/dataset.csv", keep_default_na=False)

    train_all = df[df["split"] == "train"]
    val_all = df[df["split"] == "val"]
    test_all = df[df["split"] == "test"]

    os.makedirs("results/rq3", exist_ok=True)

    results = []

    # test each task type separately
    tasks = ["technical", "writing", "qa"]

    for task in tasks:

        test_df = test_all[test_all["task_type"] == task]
        test_data = {task: test_df}

        # training without this task
        cross_train = train_all[train_all["task_type"] != task]
        cross_val = val_all[val_all["task_type"] != task]

        num_prompts = cross_train["prompt_id"].nunique()

        for seed in range(args.seeds):
            rng = np.random.RandomState(100 + seed)

            # choose the same number of prompts for comparison
            all_ids = train_all["prompt_id"].unique()
            chosen = rng.choice(all_ids, num_prompts, replace=False)

            normal_train = train_all[train_all["prompt_id"].isin(chosen)]

            for setup in ["cross", "in_domain"]:

                if setup == "cross":
                    train_df = cross_train
                    val_df = cross_val

                else:
                    train_df = normal_train
                    val_df = val_all

                print("running", task, setup, "seed", seed)

                result = run_experiment(
                    train_df,
                    val_df,
                    test_data,
                    args.model,
                    "output",
                    seed,
                    verbose=False
                )

                scores = result["tests"][task]

                row = {
                    "held_out": task,
                    "cond": setup,
                    "seed": seed,
                    "n_train": len(train_df),
                    "acc": scores["acc"],
                    "macro_f1": scores["macro_f1"],
                    "per_class_f1": scores["per_class_f1"],
                    "n_test": scores["n"]
                }

                results.append(row)

                print("accuracy:", scores["acc"])
                print("f1:", scores["macro_f1"])

                # save after each run
                filename = f"results/rq3/cross_task_{args.model}.json"

                with open(filename, "w") as f:
                    json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
