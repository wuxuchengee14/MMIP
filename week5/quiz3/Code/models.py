from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CaptionModel(nn.Module):
    def __init__(self, kind, vocab_size, dim=256, layers=2, heads=8,
                 dropout=0.2, max_length=40, grid_size=4, feature_dim=2048):
        super().__init__()
        self.kind = kind
        self.max_length = max_length
        self.layers = layers
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.projection = nn.Linear(feature_dim, dim)
        self.dropout = nn.Dropout(dropout)
        if kind == "lstm":
            self.decoder = nn.LSTM(dim, dim, num_layers=layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0)
            self.init_h = nn.Linear(dim, layers * dim)
            self.init_c = nn.Linear(dim, layers * dim)
        elif kind == "transformer":
            self.text_position = nn.Embedding(max_length, dim)
            self.image_position = nn.Parameter(torch.randn(1, grid_size**2, dim) * 0.02)
            layer = nn.TransformerDecoderLayer(dim, heads, dim_feedforward=dim * 4,
                                               dropout=dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, layers)
        else:
            raise ValueError(kind)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, features, tokens):
        memory = self.projection(features)
        text = self.dropout(self.embedding(tokens))
        if self.kind == "lstm":
            pooled = memory.mean(1)
            batch = len(tokens)
            h = self.init_h(pooled).view(batch, self.layers, -1).transpose(0, 1).contiguous()
            c = self.init_c(pooled).view(batch, self.layers, -1).transpose(0, 1).contiguous()
            hidden, _ = self.decoder(text, (h, c))
        else:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            text = text + self.text_position(positions)[None]
            causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                                device=tokens.device).triu(1)
            hidden = self.decoder(text, memory + self.image_position,
                                  tgt_mask=causal, tgt_key_padding_mask=tokens.eq(0))
        return self.output(hidden)


@torch.inference_mode()
def generate(model, features, max_length=40, beam_width=1, length_penalty=0.7):
    """Batched greedy/beam search. BOS=2, EOS=3; PAD/BOS are never generated."""
    batch, device = len(features), features.device
    tokens = torch.full((batch, 1, 1), 2, dtype=torch.long, device=device)
    scores = torch.zeros(batch, 1, device=device)
    finished = torch.zeros(batch, 1, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch, 1, dtype=torch.long, device=device)
    for _ in range(max_length - 1):
        beams, seq_length = tokens.shape[1:]
        expanded = features[:, None].expand(-1, beams, -1, -1).reshape(
            batch * beams, features.shape[1], features.shape[2])
        logits = model(expanded, tokens.reshape(batch * beams, seq_length))[:, -1].float()
        logp = F.log_softmax(logits, dim=-1).view(batch, beams, -1)
        logp[:, :, 0] = -float("inf")
        logp[:, :, 2] = -float("inf")
        # Finished beams retain their score, emitting only EOS.
        logp = logp.masked_fill(finished[:, :, None], -float("inf"))
        logp[:, :, 3] = torch.where(finished, torch.zeros_like(scores), logp[:, :, 3])
        candidates = scores[:, :, None] + logp
        vocab = logp.shape[-1]
        scores, chosen = candidates.flatten(1).topk(beam_width, dim=1)
        parents, next_tokens = chosen // vocab, chosen % vocab
        gather = parents[:, :, None].expand(-1, -1, seq_length)
        tokens = torch.cat((tokens.gather(1, gather), next_tokens[:, :, None]), dim=-1)
        previous_finished = finished.gather(1, parents)
        lengths = lengths.gather(1, parents) + (~previous_finished).long()
        finished = previous_finished | next_tokens.eq(3)
        if finished.all():
            break
    normalized = scores / ((5.0 + lengths.float()) / 6.0).pow(length_penalty)
    best = normalized.argmax(1)
    return tokens[torch.arange(batch, device=device), best]


def decode(ids, vocab):
    words = []
    for token in ids:
        token = int(token)
        if token == 3:
            break
        if token not in (0, 2):
            words.append(vocab[token])
    return " ".join(words)
