"""
Train / evaluate one CNN or LSTM run.

    python src/train.py --model cnn --mode output --seed 0
    python src/train.py --model lstm --mode both --seed 1

`run_experiment` is also imported by the RQ3 / RQ4 scripts.
"""
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


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def bucket_batches(lengths, batch_size, shuffle, rng):
    """group examples of similar length into the same batch (much faster for the LSTM)"""
    idx = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: lengths[i])
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def predict(model, ids, labels, batch_size=128):
    model.eval()
    ds = TextDataset(ids, labels)
    preds, probs = [], []
    with torch.no_grad():
        for b in bucket_batches([len(x) for x in ids], batch_size, False, None):
            X, L, _ = collate([ds[i] for i in b])
            p = torch.softmax(model(X, L), dim=1)
            probs.append((b, p.numpy()))
    P = np.zeros((len(ids), len(FAMILIES)))
    for b, p in probs:
        P[b] = p
    return P.argmax(1), P


def run_experiment(train_df, val_df, test_dfs, model_name="cnn", mode="output",
                   seed=0, ablation="none", epochs=15, patience=4, batch_size=64,
                   lr=1e-3, max_out=512, verbose=True):
    """train on train_df, early-stop on val macro-F1, evaluate on each df in test_dfs
    (dict name -> df). Returns a dict of metrics."""
    set_seed(seed)
    t0 = time.time()
    tok = lambda d: build_tokens(d, mode, max_out=max_out, ablation=ablation, seed=seed)
    tr_tok, va_tok = tok(train_df), tok(val_df)
    vocab = Vocab(tr_tok)
    enc = lambda toks: [vocab.encode(t) for t in toks]
    tr_ids, va_ids = enc(tr_tok), enc(va_tok)
    ytr = [LABEL[f] for f in train_df["llm_family"]]
    yva = np.array([LABEL[f] for f in val_df["llm_family"]])

    model = make_model(model_name, len(vocab), len(FAMILIES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ds = TextDataset(tr_ids, ytr)
    rng = random.Random(seed)
    best_f1, best_state, bad, history = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in bucket_batches([len(x) for x in tr_ids], batch_size, True, rng):
            X, L, y = collate([ds[i] for i in b])
            opt.zero_grad()
            loss = lossf(model(X, L), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        vp, _ = predict(model, va_ids, list(yva))
        vf1 = f1_score(yva, vp, average="macro")
        vacc = accuracy_score(yva, vp)
        history.append({"epoch": ep, "train_loss": tot / len(ytr), "val_acc": vacc, "val_f1": vf1})
        if verbose:
            print(f"  ep{ep:2d} loss={tot/len(ytr):.3f} val_acc={vacc:.3f} val_f1={vf1:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    res = {"model": model_name, "mode": mode, "seed": seed, "ablation": ablation,
           "vocab": len(vocab), "params": sum(p.numel() for p in model.parameters()),
           "n_train": len(train_df), "best_val_f1": best_f1, "history": history,
           "train_time_s": time.time() - t0, "tests": {}}
    for name, tdf in test_dfs.items():
        ids = enc(tok(tdf))
        y = np.array([LABEL[f] for f in tdf["llm_family"]])
        p, P = predict(model, ids, list(y))
        res["tests"][name] = {
            "n": len(y), "acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "per_class_f1": dict(zip(FAMILIES, f1_score(y, p, average=None, labels=range(6)).tolist())),
            "confusion": confusion_matrix(y, p, labels=range(6)).tolist(),
            "pred": p.tolist(),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--model", default="cnn", choices=["cnn", "lstm"])
    ap.add_argument("--mode", default="output", choices=["output", "input", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/rq12")
    args = ap.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"[{args.model}/{args.mode}/seed{args.seed}] train={len(tr)} val={len(va)} test={len(te)}")
    res = run_experiment(tr, va, {"test": te}, args.model, args.mode, args.seed)
    t = res["tests"]["test"]
    print(f"TEST acc={t['acc']:.4f} macroF1={t['macro_f1']:.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.model}_{args.mode}_s{args.seed}.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
