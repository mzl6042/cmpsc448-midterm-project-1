
import torch
import torch.nn as nn


# CNN model
class TextCNN(nn.Module):

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.convs = nn.ModuleList()

        for k in kernel_sizes:
            conv = nn.Conv1d(emb_dim, n_filters, k, padding=k // 2)
            self.convs.append(conv)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lens):

        e = self.emb(x)
        e = e.transpose(1, 2)

        mask = (x != 0).unsqueeze(1)

        features = []

        for conv in self.convs:
            h = conv(e)
            h = torch.relu(h)

            h = h[:, :, :x.size(1)]

            # don't include padding
            h = h.masked_fill(~mask, -1e4)

            pooled = h.max(dim=2).values
            features.append(pooled)

        combined = torch.cat(features, dim=1)
        combined = self.drop(combined)

        result = self.fc(combined)

        return result


# LSTM model
class BiLSTM(nn.Module):

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, dropout=0.3):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            batch_first=True,
            bidirectional=True
        )

        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden * 4, n_classes)

    def forward(self, x, lens):

        e = self.emb(x)
        e = self.drop(e)

        out, _ = self.lstm(e)

        mask = (x != 0).unsqueeze(2)

        # max pooling
        masked_out = out.masked_fill(~mask, -1e4)
        max_pool = masked_out.max(dim=1).values

        # average pooling
        total = (out * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_pool = total / count

        combined = torch.cat([max_pool, mean_pool], dim=1)

        combined = self.drop(combined)
        result = self.fc(combined)

        return result


def make_model(name, vocab_size, n_classes):

    if name == "cnn":
        model = TextCNN(vocab_size, n_classes)

    elif name == "lstm":
        model = BiLSTM(vocab_size, n_classes)

    else:
        raise ValueError(name)

    return model
