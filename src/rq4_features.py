"""
RQ4 part 1: hand-crafted linguistic / structural features.
  - per-family feature means (what is different?)
  - feature-only classifiers, by feature group (which kind of signal carries the information?)
  - permutation importance
  - most distinctive words per family (log-odds ratio with informative Dirichlet prior)
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

FAM = ["Claude", "GPT", "Google", "Llama", "Mistral", "Qwen"]
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def mattr(words, w=50):
    """moving-average type-token ratio (length-robust lexical diversity)"""
    words = [x.lower() for x in words]
    if len(words) < w:
        return len(set(words)) / max(1, len(words))
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(0, len(words) - w + 1, 5)]))


def features(text):
    words = WORD_RE.findall(text)
    nw = max(1, len(words))
    lines = text.split("\n")
    per100 = lambda c: 100.0 * c / nw
    first = text.strip()[:40].lower()
    f = {
        # --- length ---
        "len_chars": len(text),
        "len_words": len(words),
        "n_lines": len([l for l in lines if l.strip()]),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        # --- lexical ---
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "avg_sent_len": nw / max(1, len(SENT_RE.findall(text))),
        "mattr": mattr(words),
        "first_person_I": per100(sum(w == "I" for w in words)),
        "you_rate": per100(sum(w.lower() in ("you", "your") for w in words)),
        "contractions": per100(sum("'" in w for w in words)),
        # --- formatting / structure ---
        "md_headings": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "bullets": sum(bool(re.match(r"\s*[-*•+]\s", l)) for l in lines),
        "numbered": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "bold": text.count("**") // 2,
        "code_blocks": text.count("```") // 2,
        "colon_rate": per100(text.count(":")),
        "excl_rate": per100(text.count("!")),
        "question_rate": per100(text.count("?")),
        "dash_rate": per100(text.count("—") + text.count("–")),
        "paren_rate": per100(text.count("(")),
        "emoji": len(EMOJI_RE.findall(text)),
        # --- discourse habits ---
        "opens_sure_certainly": int(bool(re.match(r"(sure|certainly|absolutely|of course)", first))),
        "opens_here": int(first.startswith("here")),
        "ends_with_question": int(text.strip().endswith("?")),
        "hedge_important_note": int(bool(re.search(r"important to (note|remember)|keep in mind", text, re.I))),
        "closing_summary": int(bool(re.search(r"\b(in summary|in conclusion|overall,|to summarize)", text, re.I))),
        "offer_more_help": int(bool(re.search(r"(let me know|feel free to|hope this helps|i hope)", text, re.I))),
    }
    return f


GROUPS = {
    "length": ["len_chars", "len_words", "n_lines", "n_paragraphs"],
    "lexical": ["avg_word_len", "avg_sent_len", "mattr", "first_person_I", "you_rate", "contractions"],
    "formatting": ["md_headings", "bullets", "numbered", "bold", "code_blocks", "colon_rate",
                   "excl_rate", "question_rate", "dash_rate", "paren_rate", "emoji"],
    "discourse": ["opens_sure_certainly", "opens_here", "ends_with_question",
                  "hedge_important_note", "closing_summary", "offer_more_help"],
}


def log_odds(df, top=12, min_prompts=25):
    """Monroe et al. (2008) weighted log-odds with informative Dirichlet prior.
    We count DOCUMENT frequency (does the response use the word at all) and only keep
    words used in >= min_prompts different prompts; otherwise topic words like 'romeo'
    from a single prompt dominate the list."""
    tok = lambda t: set(w.lower() for w in WORD_RE.findall(t))
    counts = {f: Counter() for f in FAM}
    prompts_with = {}
    for f, t, pid in zip(df.llm_family, df.llm_output, df.prompt_id):
        ws = tok(t)
        counts[f].update(ws)
        for w in ws:
            prompts_with.setdefault(w, set()).add(pid)
    allowed = {w for w, s in prompts_with.items() if len(s) >= min_prompts}
    counts = {f: Counter({w: c for w, c in cnt.items() if w in allowed}) for f, cnt in counts.items()}
    prior = Counter()
    for c in counts.values():
        prior.update(c)
    a0 = sum(prior.values())
    out = {}
    for f in FAM:
        rest = Counter()
        for g in FAM:
            if g != f:
                rest.update(counts[g])
        n1, n2 = sum(counts[f].values()), sum(rest.values())
        scores = {}
        for w, aw in prior.items():
            if aw < 30:
                continue
            y1, y2 = counts[f][w], rest[w]
            d = np.log((y1 + aw) / (n1 + a0 - y1 - aw)) - np.log((y2 + aw) / (n2 + a0 - y2 - aw))
            var = 1 / (y1 + aw) + 1 / (y2 + aw)
            scores[w] = d / np.sqrt(var)
        out[f] = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]
    return out


def main():
    os.makedirs("results/rq4", exist_ok=True)
    df = pd.read_csv("data/dataset.csv", keep_default_na=False)
    F = pd.DataFrame([features(t) for t in df.llm_output])
    F["llm_family"] = df.llm_family.values
    F["split"] = df.split.values
    F.to_csv("results/rq4/features.csv", index=False)

    means = F.groupby("llm_family")[sum(GROUPS.values(), [])].mean().T
    means.to_csv("results/rq4/feature_means.csv")
    print(means.round(2))

    tr, te = F[F.split != "test"], F[F.split == "test"]
    out = {"group_results": {}}
    for name, cols in list(GROUPS.items()) + [("all", sum(GROUPS.values(), []))]:
        accs, f1s = [], []
        for seed in range(3):
            rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            rf.fit(tr[cols], tr.llm_family)
            p = rf.predict(te[cols])
            accs.append(accuracy_score(te.llm_family, p))
            f1s.append(f1_score(te.llm_family, p, average="macro"))
        out["group_results"][name] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                                      "macro_f1": float(np.mean(f1s)), "n_features": len(cols)}
        print(name, out["group_results"][name], flush=True)
    # importance with the all-features model
    cols = sum(GROUPS.values(), [])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    rf.fit(tr[cols], tr.llm_family)
    pi = permutation_importance(rf, te[cols], te.llm_family, n_repeats=5, random_state=0, n_jobs=-1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])
    out["perm_importance"] = [(c, float(v)) for c, v in imp]
    print(imp[:10])
    out["log_odds_words"] = log_odds(df[df.split != "test"])
    for f, ws in out["log_odds_words"].items():
        print(f, ws)
    json.dump(out, open("results/rq4/feature_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
