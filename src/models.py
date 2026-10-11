"""The two required classifiers: a text CNN and a bidirectional LSTM."""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """Kim (2014)-style CNN: parallel 1D convolutions with different widths,
    global max-pool over time, then a linear layer."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lens):
        e = self.emb(x).transpose(1, 2)              # B x E x T
        mask = (x != 0).unsqueeze(1)                 # B x 1 x T
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, :x.size(1)]
            h = h.masked_fill(~mask, -1e4)           # ignore padding in the max-pool
            feats.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class BiLSTM(nn.Module):
    """Bidirectional LSTM over the token sequence.
    Sentence vector = [masked max-pool ; masked mean-pool] over all hidden states.
    NOTE: we first used pack_padded_sequence, but on our CPU-only torch build a batch
    with mixed lengths took ~30s vs ~1s unpacked, so we run on padded batches instead.
    Batches are length-bucketed (see train.py) so there is very little padding,
    and padding positions are masked out of the pooling."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        out, _ = self.lstm(e)                                    # B x T x 2H
        mask = (x != 0).unsqueeze(2)
        maxp = out.masked_fill(~mask, -1e4).max(dim=1).values
        meanp = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([maxp, meanp], dim=1)))


def make_model(name, vocab_size, n_classes):
    if name == "cnn":
        return TextCNN(vocab_size, n_classes)
    if name == "lstm":
        return BiLSTM(vocab_size, n_classes)
    raise ValueError(name)
