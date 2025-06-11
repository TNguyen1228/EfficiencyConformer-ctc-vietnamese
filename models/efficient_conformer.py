from __future__ import annotations

"""Efficient Conformer encoder built on torchaudio.models.Conformer.

Keeps the same public API as `ConformerEncoder` so we can swap without touching
training code.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:
    from torchaudio.models import Conformer as TAConformer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchaudio >= 2.2 required for EfficientConformerEncoder – install with pip install torchaudio --upgrade"
    ) from e

__all__ = ["EfficientConformerEncoder"]


class EfficientConformerEncoder(nn.Module):
    """Subsample -> torchaudio Conformer stack -> optional intermediate outputs."""

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        # 2-layer conv2d front-end identical to previous encoder
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Positional encoding learnable
        self.pos_encoding = nn.Parameter(torch.randn(1, 10000, d_model) * 0.01)

        ffn_dim = d_model * ffn_expansion
        self.conformer = TAConformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=ffn_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )

        # helper for lengths
        def _freq_after_subsample(n: int) -> int:
            out = (n - 1) // 2 + 1
            out = (out - 1) // 2 + 1
            return out

        self.freq_after = _freq_after_subsample(n_mels)

        self.proj = None
        if self.freq_after > 1:
            self.proj = nn.Linear(d_model * self.freq_after, d_model)

    @staticmethod
    def _lengths_to_padding_mask(lengths: Tensor) -> Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        return (
            torch.arange(max_length, device=lengths.device).expand(batch_size, max_length)
            >= lengths.unsqueeze(1)
        )

    @staticmethod
    def _conv_out_length(length: Tensor, kernel: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> Tensor:
        """Replicate PyTorch Conv output length formula with floor division."""
        return ((length + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1

    def get_length_after_subsample(self, x_len: Tensor) -> Tensor:
        len_after = self._conv_out_length(x_len)
        len_after = self._conv_out_length(len_after)
        return len_after

    def forward(
        self,
        x: Tensor,  # (B, n_mels, T)
        x_len: Tensor,
        return_intermediate: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        # B, n_mels, T -> B,1,n_mels,T
        x = x.unsqueeze(1)
        x = self.subsample(x)  # (B, d_model, n_mels/4, T/4)
        B, D, F, TT = x.size()
        # Expect F == 1 for simplicity
        # Merge freq dim
        if self.freq_after == 1:
            x = x.squeeze(2).transpose(1, 2)  # (B,T',d_model)
        else:
            x = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), TT, D * self.freq_after)
            if self.proj is not None:
                x = self.proj(x)

        # add pos enc
        x = x + self.pos_encoding[:, : x.size(1), :]

        # lengths after subsample
        new_len = self.get_length_after_subsample(x_len)

        # Build padding mask that matches seq_len for manual layer iteration
        seq_len = x.size(1)
        padding_mask = (
            torch.arange(seq_len, device=new_len.device).expand(x.size(0), seq_len) >= new_len.unsqueeze(1)
        )

        intermediates: List[Tensor] = []
        if return_intermediate:
            # iterate manually to capture outputs every quarter
            quarter = max(1, len(self.conformer.conformer_layers) // 4)
            x_t = x.transpose(0, 1)  # (T,B,D)
            for idx, layer in enumerate(self.conformer.conformer_layers, 1):
                x_t = layer(x_t, padding_mask)
                if idx % quarter == 0:
                    intermediates.append(x_t.transpose(0, 1))
            x = x_t.transpose(0, 1)
        else:
            x, _ = self.conformer(x, new_len)

        return x, new_len, intermediates 