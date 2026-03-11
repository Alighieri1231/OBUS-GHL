"""
TemporalMeanPooling.py

Simple baseline temporal aggregator for frame embeddings.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalMeanPooling(nn.Module):
    def forward(self, x):
        weights = torch.full(
            size=(x.shape[0], x.shape[1], 1),
            fill_value=1.0 / x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
        return x.mean(dim=1), weights

