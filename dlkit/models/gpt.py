import math
from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkit.models.bases import ModelConfig


class SelfAttention(nn.Module):
    """Self-Attention with an optional causal mask."""

    def __init__(
        self,
        d_model=512,
        n_head=4,
        dropout=0.5,
        autoregressive=False,
        max_positions=256,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        # linear projection for K, Q, V all
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model)
        # regularizations
        self.attn_dropout = nn.Dropout(p=dropout)
        self.residual_dropout = nn.Dropout(p=dropout)
        self.autoregressive = autoregressive
        # causal mask to ensure the autoregressiveness.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_positions, max_positions))[None, None, :, :],
        )

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape  # batch size, sequence length, embedding size

        q, k, v = self.in_proj(x).split(self.d_model, dim=2)
        # print(f"shapes: {q.shape=}, {k.shape=}, {v.shape=}")
        q = einops.rearrange(q, "b s (head dk) -> head b s dk", head=self.n_head)
        k = einops.rearrange(k, "b t (head dk) -> head b t dk", head=self.n_head)
        v = einops.rearrange(v, "b t (head dv) -> head b t dv", head=self.n_head)
        # print(f"shapes: {q.shape=}, {k.shape=}, {v.shape=}")
        attn = torch.einsum("hbsd,hbtd->hbst", [q, k]) / math.sqrt(q.shape[-1])
        if self.autoregressive is True:
            # enforce autoregressiveness.
            attn = attn.masked_fill(
                self.mask[:, :, :S, :S] == 0, float("-inf")
            )  # -inf becomes 0 after softmax
        attn = torch.softmax(attn, dim=3)

        output = torch.einsum("hbst,hbtd->hbsd", [attn, v])
        # print(f"{output.shape=}")
        output = einops.rearrange(output, "h b s d -> b s (h d)")
        # out projection
        y = self.residual_dropout(self.out_proj(output))
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.5, max_positions=256):
        super().__init__()
        assert d_model % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(d_model, d_model)
        # causal mask to ensure that attention is only applied
        # to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_positions, max_positions)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch
        # and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention;
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class SABlock(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_head=4,
        dropout=0.5,
        autoregressive=False,
        max_positions=256,
        dim_feedforward=1024,
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model, bias=False)
        self.ln2 = LayerNorm(d_model, bias=False)
        self.attn = SelfAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            autoregressive=autoregressive,
            max_positions=max_positions,
        )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            LayerNorm(dim_feedforward, bias=False),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            LayerNorm(d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """forward batch first input x"""
        x = x + self.attn(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, n_features, bias: bool = False) -> None:
        super().__init__()
        self.d = n_features
        self.a_2 = nn.Parameter(torch.ones(self.d))
        self.b_2 = nn.Parameter(torch.zeros(self.d)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.d,), self.a_2, self.b_2, 1e-6)


@dataclass
class GPTConfig(ModelConfig):
    d_in: int = None
    d_out: int = None
    d_model: int = 512
    n_head: int = 4
    n_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_positions: int = 256


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.d_in is not None
        assert config.d_out is not None
        self.config = config
        self.in_proj = nn.Linear(config.d_in, config.d_model)
        self.in_ln = nn.LayerNorm(config.d_model)
        # learnable position embedding
        self.pe = nn.Embedding(config.max_positions, config.d_model)
        self.in_drop = nn.Dropout(config.dropout)

        self.attn_layers = nn.ModuleList(
            SABlock(
                d_model=config.d_model,
                n_head=config.n_head,
                dropout=config.dropout,
                autoregressive=True,
                max_positions=config.max_positions,
                dim_feedforward=config.dim_feedforward,
            )
            for _ in range(config.n_layers)
        )
        self.out_ln = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_out)
        self.register_buffer("positions", torch.arange(config.max_positions)[None, :])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """feedforward.
        Args:
            x: torch.Tensor in shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: result Tensor in shape: (batch_size, seq_len, d_out).
        """
        x_proj = self.in_ln(self.in_proj(x))
        # x_position_embed = self.pe(self.positions[:, :x_proj.shape[1]]).to(x_proj.device)
        # x = self.in_drop(x_position_embed + x_proj)
        o = self.in_drop(x_proj)

        for layer in self.attn_layers:
            o = layer(o)

        o = self.out_proj(self.out_ln(o))
        return o
