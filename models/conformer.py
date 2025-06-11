from __future__ import annotations

"""Lightweight Conformer encoder (only what is needed for our ASR training).
The implementation follows the architecture described in
https://arxiv.org/abs/2005.08100 but simplifies some components to keep code short
and dependency-free. It is NOT numerically identical to ESPnet/WeNet variants but
works well as a drop-in replacement for the prior AudioEncoder.

Notation
--------
B : batch size
T : time steps (frames)
D : model dimension (``d_model``)

Input tensor has shape ``(B, n_mels, T)`` like the original ``AudioEncoder``.
The first convolutional subsampler reduces the time resolution by a factor of 4
(2×Conv2d with stride 2). The sequence is then transposed to ``(B, T', D)`` and
fed through a stack of ``ConformerBlock``.

The encoder returns
    encoded: Tensor  ``(B, T', D)``
    enc_len: Tensor  length after subsampling for each example
    intermediates: list[Tensor]  encoder outputs after every quarter of the
        blocks (used for auxiliary CTC loss during training).
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = [
    "ConformerBlock",
    "ConformerEncoder",
]


class _FeedForwardModule(nn.Module):
    """Position-wise feed-forward module with Factorized FF (Macaron style)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class _DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution used in Conformer."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            2 * d_model,
            2 * d_model,
            kernel_size=kernel_size,
            groups=2 * d_model,
            padding=(kernel_size - 1) // 2,
        )
        self.batch_norm = nn.BatchNorm1d(2 * d_model)
        self.pointwise_conv2 = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GLU(dim=1)

    def forward(self, x: Tensor) -> Tensor:  # x (B, T, D)
        # convert to (B, D, T)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(d_model, ff_expansion, dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv_module = _DepthwiseConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = _FeedForwardModule(d_model, ff_expansion, dropout)

        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_mha = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # Macaron FFN module (1/2 scale)
        residual = x
        x = self.norm_ffn1(x)
        x = residual + 0.5 * self.dropout(self.ffn1(x))

        # Self-attention module
        residual = x
        x = self.norm_mha(x)
        attn_out, _ = self.self_attn(
            x, x, x, need_weights=False, key_padding_mask=src_key_padding_mask
        )
        x = residual + self.dropout(attn_out)

        # Convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.conv_module(x)

        # Feed-forward module (1/2 scale)
        residual = x
        x = self.norm_ffn2(x)
        x = residual + 0.5 * self.dropout(self.ffn2(x))

        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # compute frequency dimension after two stride-2 convs on frequency axis
        def _freq_after_subsample(n_freq: int) -> int:
            # conv formula with k=3, p=1, s=2 applied twice
            out = (n_freq - 1) // 2 + 1
            out = (out - 1) // 2 + 1
            return out

        freq_after = _freq_after_subsample(n_mels)

        self.pos_encoding = nn.Parameter(torch.randn(1, 10000, d_model) * 0.01)

        # If freq_after > 1 we need a linear projection from d_model*freq_after -> d_model
        self.proj = None
        if freq_after > 1:
            self.proj = nn.Linear(d_model * freq_after, d_model)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    n_heads,
                    ff_expansion=4,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)

    def get_length_after_subsample(self, x_len: Tensor) -> Tensor:
        # two Conv2d with stride 2 on time dimension
        return ((x_len - 1) // 2 - 1) // 2 + 1  # approx; assumes padding preserves length

    def forward(
        self,
        x: Tensor,  # (B, n_mels, T)
        x_len: Tensor,
        return_intermediate: bool = False,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        B, C, T = x.size()
        assert C == x.size(1)

        # Add channel dim for Conv2d
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.subsample(x)  # (B, d_model, n_mels/4, T/4)
        # merge feature and channel dims -> time major
        B, D, F, TT = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, TT, D * F)  # (B, T', D*F)

        # positional encoding (simple learned) – slice to needed length
        pos_enc = self.pos_encoding[:, : x.size(1), :]
        x = x + pos_enc

        intermediates: List[Tensor] = []

        layer_quarter = max(1, len(self.layers) // 4)
        for idx, layer in enumerate(self.layers, 1):
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            if return_intermediate and idx % layer_quarter == 0:
                intermediates.append(self.norm_out(x))

        x = self.norm_out(x)
        enc_len = self.get_length_after_subsample(x_len)

        # Project to d_model if needed
        if self.proj is not None:
            x = self.proj(x)

        return x, enc_len, intermediates 