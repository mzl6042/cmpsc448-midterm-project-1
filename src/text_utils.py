"""Tokenization, vocabulary and input construction shared by all experiments."""
import random
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"

# keep words, numbers, and every punctuation/markdown symbol as its own token.
# newlines are kept as a token too because formatting (lists, headings) is a big part of style.
TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[a-z]+)?|\d+|[^\w\s]")


def tokenize(text):
    toks = TOKEN_RE.findall(str(text))
    return [NL if t == "\n" else t for t in toks]


class Vocab:
    def __init__(self, token_lists, max_size=30000, min_freq=2):
        cnt = Counter()
        for toks in token_lists:
            cnt.update(toks)
        self.itos = [PAD, UNK, SEP, NL]
        for tok, c in cnt.most_common():
            if len(self.itos) >= max_size or c < min_freq:
                break
            if tok not in (PAD, UNK, SEP, NL):
                self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]


# ---------------- text "ablations" used for RQ4 ----------------
MD_RE = re.compile(r"[#*_`>|~]+")
LIST_RE = re.compile(r"^\s*(?:[-•+]|\d+[.)])\s+", re.M)


def strip_formatting(text):
    """remove markdown + list markers + line structure -> plain paragraph"""
    text = LIST_RE.sub("", text)
    text = MD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_ablation(tokens, text, name, rng):
    if name == "none":
        return tokens
    if name == "no_format":
        return tokenize(strip_formatting(text))
    if name == "first50":           # only the opening of the answer
        return tokens[:50]
    if name == "shuffle":           # destroy word order, keep the bag of words
        t = list(tokens)
        rng.shuffle(t)
        return t
    if name == "lower_nopunct":     # only lowercase words, no punctuation/format
        return [t.lower() for t in tokens if t[0].isalnum()]
    raise ValueError(name)


def build_tokens(df, mode, max_in=128, max_out=512, ablation="none", seed=0):
    """mode: 'output' | 'input' | 'both'"""
    rng = random.Random(seed)
    out = []
    for inp, outp in zip(df["llm_input"], df["llm_output"]):
        ti = tokenize(inp)[:max_in]
        to = apply_ablation(tokenize(outp), outp, ablation, rng)[:max_out]
        if mode == "output":
            toks = to
        elif mode == "input":
            toks = ti
        else:
            toks = ti + [SEP] + to
        out.append(toks if toks else [UNK])
    return out


class TextDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    lens = torch.tensor([len(x) for x, _ in batch])
    X = torch.zeros(len(batch), int(lens.max()), dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        X[i, :len(x)] = torch.tensor(x)
    y = torch.tensor([lab for _, lab in batch])
    return X, lens, y
