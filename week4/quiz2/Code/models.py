import torch
from torch import nn


class SentimentModel(nn.Module):
    def __init__(self, kind, vocab_size, embedding_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind]
        self.recurrent = recurrent(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, tokens, lengths):
        output, _ = self.recurrent(self.embedding(tokens))
        # Unidirectional recurrence: padding after the last real token cannot
        # affect this gathered state. Avoids slower packed-sequence CUDA kernels.
        state = output[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.head(state)
