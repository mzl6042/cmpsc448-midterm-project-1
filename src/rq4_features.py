
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
    words = [word.lower() for word in words]

    if len(words) < w:
        return len(set(words)) / max(1, len(words))

    scores = []

    for i in range(0, len(words) - w + 1, 5):
        part = words[i:i + w]
        score = len(set(part)) / w
        scores.append(score)

    return float(np.mean(scores))


def features(text):
    words = WORD_RE.findall(text)
    nw = max(1, len(words))
    lines = text.split("\n")
    first = text.strip()[:40].lower()

    # count paragraphs and sentences
    paragraphs = re.split(r"\n\s*\n", text)
    n_paragraphs = sum(1 for p in paragraphs if p.strip())

    if len(words) > 0:
        avg_word_len = np.mean([len(w) for w in words])
    else:
        avg_word_len = 0

    sentences = len(SENT_RE.findall(text))

    def per100(count):
        return 100.0 * count / nw

    f = {
        "len_chars": len(text),
        "len_words": len(words),
        "n_lines": sum(1 for l in lines if l.strip()),
        "n_paragraphs": n_paragraphs,

        "avg_word_len": avg_word_len,
        "avg_sent_len": nw / max(1, sentences),
        "mattr": mattr(words),
        "first_person_I": per100(sum(w == "I" for w in words)),
        "you_rate": per100(sum(w.lower() in ("you", "your") for w in words)),
        "contractions": per100(sum("'" in w for w in words)),

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

        "opens_sure_certainly": int(bool(re.match(
            r"(sure|certainly|absolutely|of course)", first
        ))),
        "opens_here": int(first.startswith("here")),
        "ends_with_question": int(text.strip().endswith("?")),
        "hedge_important_note": int(bool(re.search(
            r"important to (note|remember)|keep in mind", text, re.I
        ))),
        "closing_summary": int(bool(re.search(
            r"\b(in summary|in conclusion|overall,|to summarize)", text, re.I
        ))),
        "offer_more_help": int(bool(re.search(
            r"(let me know|feel free to|hope this helps|i hope)", text, re.I
        )))
    }

    return f


# different features to compare
GROUPS = {
    "length": [
        "len_chars", "len_words", "n_lines", "n_paragraphs"
    ],
    "lexical": [
        "avg_word_len", "avg_sent_len", "mattr",
        "first_person_I", "you_rate", "contractions"
    ],
    "formatting": [
        "md_headings", "bullets", "numbered", "bold",
        "code_blocks", "colon_rate", "excl_rate",
        "question_rate", "dash_rate", "paren_rate", "emoji"
    ],
    "discourse": [
        "opens_sure_certainly", "opens_here",
        "ends_with_question", "hedge_important_note",
        "closing_summary", "offer_more_help"
    ]
}


def log_odds(df, top=12, min_prompts=25):
    # compare words that show up more often for each family
    # using weighted log odds with a prior (Monroe et al., 2008)

    counts = {}
    for family in FAM:
        counts[family] = Counter()

    prompts_with = {}

    for family, text, pid in zip(df.llm_family, df.llm_output, df.prompt_id):
        words = set(w.lower() for w in WORD_RE.findall(text))
        counts[family].update(words)

        for word in words:
            if word not in prompts_with:
                prompts_with[word] = set()

            prompts_with[word].add(pid)

    allowed = set()

    for word, ids in prompts_with.items():
        if len(ids) >= min_prompts:
            allowed.add(word)

    # skip words that only show up for a few prompts
    for family in FAM:
        new_counts = Counter()

        for word, count in counts[family].items():
            if word in allowed:
                new_counts[word] = count

        counts[family] = new_counts

    prior = Counter()

    for family in FAM:
        prior.update(counts[family])

    a0 = sum(prior.values())
    results = {}

    for family in FAM:
        rest = Counter()

        for other in FAM:
            if other != family:
                rest.update(counts[other])

        n1 = sum(counts[family].values())
        n2 = sum(rest.values())

        scores = {}

        for word, aw in prior.items():
            if aw < 30:
                continue

            y1 = counts[family][word]
            y2 = rest[word]

            first = np.log((y1 + aw) / (n1 + a0 - y1 - aw))
            second = np.log((y2 + aw) / (n2 + a0 - y2 - aw))

            difference = first - second
            variance = 1 / (y1 + aw) + 1 / (y2 + aw)

            scores[word] = difference / np.sqrt(variance)

        sorted_words = sorted(scores.items(), key=lambda x: -x[1])

        results[family] = []

        for word, score in sorted_words[:top]:
            results[family].append(word)

    return results


def main():
    os.makedirs("results/rq4", exist_ok=True)

    df = pd.read_csv("data/dataset.csv", keep_default_na=False)

    # get the features for each response
    feature_rows = []

    for text in df["llm_output"]:
        feature_rows.append(features(text))

    F = pd.DataFrame(feature_rows)

    F["llm_family"] = df["llm_family"].values
    F["split"] = df["split"].values

    F.to_csv("results/rq4/features.csv", index=False)

    all_cols = []

    for cols in GROUPS.values():
        all_cols.extend(cols)

    means = F.groupby("llm_family")[all_cols].mean().T
    means.to_csv("results/rq4/feature_means.csv")

    print(means.round(2))

    train_df = F[F["split"] != "test"]
    test_df = F[F["split"] == "test"]

    results = {"group_results": {}}

    groups = list(GROUPS.items())
    groups.append(("all", all_cols))

    # train using each feature group
    for name, cols in groups:
        accs = []
        f1s = []

        for seed in range(3):
            rf = RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            )

            rf.fit(train_df[cols], train_df["llm_family"])

            predictions = rf.predict(test_df[cols])

            acc = accuracy_score(test_df["llm_family"], predictions)
            f1 = f1_score(test_df["llm_family"], predictions, average="macro")

            accs.append(acc)
            f1s.append(f1)

        results["group_results"][name] = {
            "acc": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)),
            "n_features": len(cols)
        }

        print(name, results["group_results"][name], flush=True)

    # see which features matter the most
    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1
    )

    rf.fit(train_df[all_cols], train_df["llm_family"])

    pi = permutation_importance(
        rf,
        test_df[all_cols],
        test_df["llm_family"],
        n_repeats=5,
        random_state=0,
        n_jobs=-1
    )

    importance = list(zip(all_cols, pi.importances_mean))
    importance.sort(key=lambda x: -x[1])

    results["perm_importance"] = []

    for col, value in importance:
        results["perm_importance"].append((col, float(value)))

    print(importance[:10])

    # check a few writing habits for each model
    habits = {
        "curly_apostrophe": df.llm_output.str.contains("’"),

        "e.g.": df.llm_output.str.contains(
            r"e\.g\.", regex=True
        ),

        "great_question_opener": df.llm_output.str.match(
            r"\s*(what a|that's a|great|excellent) .{0,20}question",
            case=False
        ),

        "starts_with_list_or_bold": df.llm_output.str.match(
            r"\s*(\d+\.|\*\*|-|#)"
        ),

        "very_short_<20_words": df.llm_output.str.split().str.len() < 20,

        "knowledge_cutoff_mention": df.llm_output.str.contains(
            r"(as of my|knowledge cutoff|last (trained|update))",
            case=False,
            regex=True
        )
    }

    H = pd.DataFrame(habits)
    H = H.astype(float)
    H["llm_model"] = df["llm_model"].values

    habit_means = H.groupby("llm_model").mean()
    habit_means = (habit_means * 100).round(1)

    habit_means.to_csv("results/rq4/habits_per_model.csv")
    print(habit_means)

    # find words that are more common for each family
    train_data = df[df["split"] != "test"]
    results["log_odds_words"] = log_odds(train_data)

    for family, words in results["log_odds_words"].items():
        print(family, words)

    with open("results/rq4/feature_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
