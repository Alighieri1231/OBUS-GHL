"""
TemporalLSTM.py

Sequence aggregator for frame embeddings.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=effective_dropout,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        output, (hidden, _cell) = self.rnn(x)
        if self.rnn.bidirectional:
            context = nn.functional.relu(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        else:
            context = hidden[-1]
        return context, output
