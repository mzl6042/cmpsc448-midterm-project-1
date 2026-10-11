"""Aggregate every result file into results/summary.json and draw the report figures."""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

FAM = ["Claude", "GPT", "Google", "Llama", "Mistral", "Qwen"]
C = {"cnn": "#0072B2", "lstm": "#D55E00", "tfidf": "#999999", "cross": "#D55E00", "in_domain": "#0072B2"}
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
S = {}


def ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


# ---------------- RQ1 / RQ2 ----------------
runs = defaultdict(list)
for f in glob.glob("results/rq12/*.json"):
    r = json.load(open(f))
    runs[(r["model"], r["mode"])].append(r)
S["rq12"] = {}
for (m, mode), rs in sorted(runs.items()):
    acc = ms([r["tests"]["test"]["acc"] for r in rs])
    f1 = ms([r["tests"]["test"]["macro_f1"] for r in rs])
    pcf = {k: float(np.mean([r["tests"]["test"]["per_class_f1"][k] for r in rs])) for k in FAM}
    S["rq12"][f"{m}_{mode}"] = {
        "acc": acc[0], "acc_std": acc[1], "f1": f1[0], "f1_std": f1[1], "n_seeds": len(rs),
        "per_class_f1": pcf, "params": rs[0]["params"], "vocab": rs[0]["vocab"],
        "epochs_run": [len(r["history"]) for r in rs],
        "best_epoch": [int(np.argmax([h["val_f1"] for h in r["history"]]) + 1) for r in rs],
        "time_min": float(np.mean([r["train_time_s"] for r in rs]) / 60)}
if os.path.exists("results/baseline_tfidf.json"):
    S["tfidf"] = json.load(open("results/baseline_tfidf.json"))

modes = ["output", "input", "both"]
mode_lbl = {"output": "Output only", "input": "Input only", "both": "Input + Output"}
fig, ax = plt.subplots(figsize=(6.2, 2.8))
w = 0.26
for i, m in enumerate(["cnn", "lstm", "tfidf"]):
    vals, errs = [], []
    for mode in modes:
        if m == "tfidf":
            vals.append(S.get("tfidf", {}).get(mode, {}).get("acc", np.nan)); errs.append(0)
        else:
            d = S["rq12"].get(f"{m}_{mode}", {})
            vals.append(d.get("acc", np.nan)); errs.append(d.get("acc_std", 0))
    x = np.arange(3) + (i - 1) * w
    ax.bar(x, vals, w, yerr=errs, color=C[m], capsize=2,
           label={"cnn": "CNN", "lstm": "BiLSTM (RNN)", "tfidf": "TF-IDF + LogReg (reference)"}[m])
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
ax.axhline(1 / 6, ls="--", c="k", lw=0.8)
ax.text(2.45, 1 / 6 + 0.01, "chance", fontsize=7, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([mode_lbl[m] for m in modes])
ax.set_ylabel("Test accuracy (6 families)"); ax.set_ylim(0, 0.8)
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.savefig(f"{FIG}/rq12_accuracy.png"); plt.close(fig)

# confusion matrices (summed over seeds, row-normalised)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
for ax, m in zip(axes, ["cnn", "lstm"]):
    rs = runs.get((m, "output"), [])
    if not rs:
        continue
    cm = np.sum([np.array(r["tests"]["test"]["confusion"]) for r in rs], axis=0)
    cmn = cm / cm.sum(1, keepdims=True)
    S["rq12"][f"{m}_output"]["confusion_norm"] = cmn.round(3).tolist()
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(FAM, rotation=45, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels(FAM, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title({"cnn": "CNN", "lstm": "BiLSTM"}[m] + " (output only)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/rq1_confusion.png"); plt.close(fig)

# training curves
fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
for ax, m in zip(axes, ["cnn", "lstm"]):
    for r in runs.get((m, "output"), []):
        h = pd.DataFrame(r["history"])
        ax.plot(h.epoch, h.train_loss, c="#999999", lw=1)
        ax2 = ax.twinx() if r["seed"] == 0 else ax2
        ax2.plot(h.epoch, h.val_f1, c=C[m], lw=1)
        ax2.set_ylim(0.2, 0.7); ax2.set_ylabel("val macro-F1", color=C[m])
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color="#666666")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title({"cnn": "CNN", "lstm": "BiLSTM"}[m] + " (output only, all seeds)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/training_curves.png"); plt.close(fig)

# accuracy per individual model (are the 2 members of a family equally recognisable?)
test_df = pd.read_csv("data/dataset.csv", keep_default_na=False)
test_df = test_df[test_df.split == "test"].reset_index(drop=True)
S["per_model_acc"] = {}
for m in ["cnn", "lstm"]:
    rs = runs.get((m, "output"), [])
    if rs:
        accs = defaultdict(list)
        for r in rs:
            pred = np.array(r["tests"]["test"]["pred"])
            ok = np.array(FAM)[pred] == test_df.llm_family.values
            for mod, v in pd.Series(ok).groupby(test_df.llm_model.values).mean().items():
                accs[mod].append(v)
        S["per_model_acc"][m] = {k: float(np.mean(v)) for k, v in accs.items()}

# per-class F1
fig, ax = plt.subplots(figsize=(6.2, 2.4))
for i, m in enumerate(["cnn", "lstm"]):
    d = S["rq12"].get(f"{m}_output")
    if d:
        ax.bar(np.arange(6) + (i - 0.5) * 0.38, [d["per_class_f1"][k] for k in FAM], 0.38,
               color=C[m], label={"cnn": "CNN", "lstm": "BiLSTM"}[m])
ax.set_xticks(range(6)); ax.set_xticklabels(FAM); ax.set_ylabel("Test F1 (output only)")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7)
fig.savefig(f"{FIG}/rq1_per_class_f1.png"); plt.close(fig)

# ---------------- RQ2 bias ----------------
if os.path.exists("results/rq2_bias/input_only_bias.json"):
    b = pd.DataFrame(json.load(open("results/rq2_bias/input_only_bias.json")))
    g = b.groupby("p_bias").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                f1=("macro_f1", "mean"), maj=("majority_baseline", "mean"))
    S["rq2_bias"] = g.reset_index().to_dict("records")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.errorbar(g.index, g.acc, yerr=g.acc_std, marker="o", c=C["cnn"], capsize=2,
                label="CNN, input only")
    ax.plot(g.index, g.maj, "k--", lw=0.8, label="majority class")
    ax.set_xlabel("P(family = family linked to prompt source)")
    ax.set_ylabel("Test accuracy"); ax.legend(frameon=False, fontsize=7); ax.set_ylim(0, 0.8)
    fig.savefig(f"{FIG}/rq2_bias.png"); plt.close(fig)

# ---------------- RQ3 ----------------
f3 = "results/rq3/cross_task_cnn.json"
if os.path.exists(f3):
    r3 = pd.DataFrame(json.load(open(f3)))
    g = r3.groupby(["held_out", "cond"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                            f1=("macro_f1", "mean"), n_train=("n_train", "mean"),
                                            n_test=("n_test", "mean")).reset_index()
    S["rq3"] = g.to_dict("records")
    pc = defaultdict(dict)
    for (T, cond), sub in r3.groupby(["held_out", "cond"]):
        pc[T][cond] = {k: float(np.mean([p[k] for p in sub.per_class_f1])) for k in FAM}
    S["rq3_per_class"] = pc
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    Ts = ["technical", "writing", "qa"]
    for i, cond in enumerate(["in_domain", "cross"]):
        sub = g[g.cond == cond].set_index("held_out").reindex(Ts)
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, sub.acc, 0.38, yerr=sub.acc_std.fillna(0),
               capsize=2, color=C[cond],
               label={"in_domain": "train includes task type", "cross": "task type never seen"}[cond])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Technical\n(code/math)", "Writing", "General QA"])
    ax.set_ylabel("Test accuracy (CNN)"); ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=7)
    ax.axhline(1 / 6, ls="--", c="k", lw=0.8)
    fig.savefig(f"{FIG}/rq3_cross_task.png"); plt.close(fig)

# ---------------- RQ4 ----------------
if os.path.exists("results/rq4/feature_results.json"):
    S["rq4_features"] = json.load(open("results/rq4/feature_results.json"))
    fm = pd.read_csv("results/rq4/feature_means.csv", index_col=0)
    S["rq4_feature_means"] = fm.round(3).to_dict()
    show = [("len_words", "words / response"), ("n_paragraphs", "paragraphs"),
            ("bold", "**bold** spans"), ("bullets", "bullet lines"),
            ("numbered", "numbered-list lines"), ("md_headings", "# headings"),
            ("opens_here", "starts with \"Here\" (rate)"), ("excl_rate", "! per 100 words")]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.4))
    for ax, (col, lbl) in zip(axes.flat, show):
        ax.bar(range(6), fm.loc[col, FAM], color="#56B4E9")
        ax.set_title(lbl, fontsize=8); ax.set_xticks(range(6))
        ax.set_xticklabels(FAM, rotation=60, fontsize=6.5); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/rq4_feature_means.png"); plt.close(fig)

f4 = "results/rq4/ablation_cnn.json"
if os.path.exists(f4):
    a = pd.DataFrame(json.load(open(f4)))
    full = S["rq12"].get("cnn_output", {})
    rows = [{"ablation": "none", "acc": full.get("acc"), "acc_std": full.get("acc_std"),
             "f1": full.get("f1"), "n": full.get("n_seeds")}]
    for ab, sub in a.groupby("ablation"):
        rows.append({"ablation": ab, "acc": sub.acc.mean(), "acc_std": sub.acc.std(ddof=0),
                     "f1": sub.macro_f1.mean(), "n": len(sub)})
    S["rq4_ablation"] = rows
    order = ["none", "no_format", "lower_nopunct", "shuffle", "first50"]
    lbl = {"none": "original", "no_format": "markdown &\nlines removed",
           "lower_nopunct": "lowercase words\nonly", "shuffle": "words\nshuffled",
           "first50": "first 50\ntokens only"}
    d = pd.DataFrame(rows).set_index("ablation").reindex(order).dropna(subset=["acc"])
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(range(len(d)), d.acc, yerr=d.acc_std, capsize=2,
           color=["#0072B2"] + ["#E69F00"] * (len(d) - 1))
    for i, v in enumerate(d.acc):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([lbl[i] for i in d.index], fontsize=7)
    ax.set_ylabel("Test accuracy (CNN, output)"); ax.set_ylim(0, 0.8)
    ax.axhline(1 / 6, ls="--", c="k", lw=0.8)
    fig.savefig(f"{FIG}/rq4_ablation.png"); plt.close(fig)

json.dump(S, open("results/summary.json", "w"), indent=1)
print(json.dumps({k: {kk: (round(vv["acc"], 4), round(vv["acc_std"], 4)) for kk, vv in v.items()}
                  for k, v in S.items() if k == "rq12"}, indent=1))
