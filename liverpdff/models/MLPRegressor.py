"""
MLPRegressor.py

Simple regression head for PDFF experiments.
"""
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256]

        layers = []
        in_features = input_dim
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

