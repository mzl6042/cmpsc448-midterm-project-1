
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import make_model
from text_utils import TextDataset, Vocab, build_tokens, collate


FAMILIES = ["Claude", "GPT", "Google", "Llama", "Mistral", "Qwen"]
LABEL = {f: i for i, f in enumerate(FAMILIES)}

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bucket_batches(lengths, batch_size, shuffle, rng):
    # put similar length texts together so lstm isn't too slow
    order = list(range(len(lengths)))

    if shuffle:
        rng.shuffle(order)

    batches = []
    chunk_size = batch_size * 50

    for i in range(0, len(order), chunk_size):
        part = order[i:i + chunk_size]
        part.sort(key=lambda x: lengths[x])

        for j in range(0, len(part), batch_size):
            batch = part[j:j + batch_size]
            batches.append(batch)

    if shuffle:
        rng.shuffle(batches)

    return batches


def predict(model, ids, labels, batch_size=128):
    model.eval()
    dataset = TextDataset(ids, labels)

    probs = np.zeros((len(ids), len(FAMILIES)))
    lengths = [len(x) for x in ids]

    with torch.no_grad():
        batches = bucket_batches(lengths, batch_size, False, None)

        for batch in batches:
            X, L, y = collate([dataset[i] for i in batch])

            output = model(X, L)
            p = torch.softmax(output, dim=1)

            probs[batch] = p.numpy()

    predictions = probs.argmax(axis=1)

    return predictions, probs


def run_experiment(train_df, val_df, test_dfs, model_name="cnn", mode="output",
                   seed=0, ablation="none", epochs=15, patience=4, batch_size=64,
                   lr=1e-3, max_out=512, verbose=True):

    set_seed(seed)
    start = time.time()

    # prepare the text
    train_tokens = build_tokens(train_df, mode, max_out=max_out,
                                ablation=ablation, seed=seed)
    val_tokens = build_tokens(val_df, mode, max_out=max_out,
                              ablation=ablation, seed=seed)

    vocab = Vocab(train_tokens)

    train_ids = []
    for tokens in train_tokens:
        train_ids.append(vocab.encode(tokens))

    val_ids = []
    for tokens in val_tokens:
        val_ids.append(vocab.encode(tokens))

    train_y = [LABEL[f] for f in train_df["llm_family"]]
    val_y = np.array([LABEL[f] for f in val_df["llm_family"]])

    model = make_model(model_name, len(vocab), len(FAMILIES))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    dataset = TextDataset(train_ids, train_y)
    rng = random.Random(seed)

    best_f1 = -1
    best_state = None
    bad = 0
    history = []

    # train the model
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        lengths = [len(x) for x in train_ids]
        batches = bucket_batches(lengths, batch_size, True, rng)

        for batch in batches:
            X, L, y = collate([dataset[i] for i in batch])

            optimizer.zero_grad()

            output = model(X, L)
            loss = loss_func(output, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(batch)

        # check validation scores
        val_pred, _ = predict(model, val_ids, list(val_y))

        val_f1 = f1_score(val_y, val_pred, average="macro")
        val_acc = accuracy_score(val_y, val_pred)

        avg_loss = total_loss / len(train_y)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        if verbose:
            print(f"epoch {epoch}: loss={avg_loss:.3f} "
                  f"acc={val_acc:.3f} f1={val_f1:.3f} "
                  f"time={time.time()-start:.0f}s", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad = 0

            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.clone()
        else:
            bad += 1

            if bad >= patience:
                break

    # use the best version from validation
    model.load_state_dict(best_state)

    results = {
        "model": model_name,
        "mode": mode,
        "seed": seed,
        "ablation": ablation,
        "vocab": len(vocab),
        "params": sum(p.numel() for p in model.parameters()),
        "n_train": len(train_df),
        "best_val_f1": best_f1,
        "history": history,
        "train_time_s": time.time() - start,
        "tests": {}
    }

    # test on each dataset
    for name, test_df in test_dfs.items():
        test_tokens = build_tokens(test_df, mode, max_out=max_out,
                                   ablation=ablation, seed=seed)

        test_ids = []
        for tokens in test_tokens:
            test_ids.append(vocab.encode(tokens))

        test_y = np.array([LABEL[f] for f in test_df["llm_family"]])

        predictions, probs = predict(model, test_ids, list(test_y))

        acc = accuracy_score(test_y, predictions)
        macro_f1 = f1_score(test_y, predictions, average="macro")

        scores = f1_score(test_y, predictions, average=None,
                          labels=range(6))

        class_f1 = {}
        for i in range(len(FAMILIES)):
            class_f1[FAMILIES[i]] = scores[i].item()

        conf = confusion_matrix(test_y, predictions,
                                labels=range(6))

        results["tests"][name] = {
            "n": len(test_y),
            "acc": acc,
            "macro_f1": macro_f1,
            "per_class_f1": class_f1,
            "confusion": conf.tolist(),
            "pred": predictions.tolist()
        }

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--model", default="cnn", choices=["cnn", "lstm"])
    parser.add_argument("--mode", default="output",
                        choices=["output", "input", "both"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="results/rq12")

    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"model: {args.model}, mode: {args.mode}, seed: {args.seed}")
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    results = run_experiment(
        train_df,
        val_df,
        {"test": test_df},
        args.model,
        args.mode,
        args.seed
    )

    test_results = results["tests"]["test"]

    print(f"test accuracy: {test_results['acc']:.4f}")
    print(f"test macro f1: {test_results['macro_f1']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    filename = f"{args.model}_{args.mode}_s{args.seed}.json"
    path = os.path.join(args.out_dir, filename)

    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()