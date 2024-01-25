from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dlkit.models.bases import ModelConfig
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, n_features, bias) -> None:
        super().__init__()
        self.d = n_features
        self.a_2 = nn.Parameter(torch.ones(self.d))
        self.b_2 = nn.Parameter(torch.zeros(self.d)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.d,), self.a_2, self.b_2, 1e-6)


@dataclass
class LSTMConfig(ModelConfig):
    d_in: int = None
    d_out: int = None
    d_model: int = 512
    n_layers: int = 2
    dropout: float = 0.1


class LSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        assert config.d_in is not None
        assert config.d_out is not None
        self.in_proj = nn.Linear(config.d_in, config.d_model)
        self.in_ln = LayerNorm(config.d_model, bias=False)

        self.encoder = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False,
        )

        self.out_ln = LayerNorm(config.d_model, bias=False)
        self.out_proj = torch.nn.Linear(config.d_model, config.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.in_proj(x)
        o = self.in_ln(o)
        o, _ = self.encoder(o)
        o = self.out_proj(self.out_ln(o))
        return o
