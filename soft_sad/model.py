"""Tiny GRU classifier for frame-level speech probability.

Architecture (per the paper):
    BatchNorm(F)  →  GRU(F → H, num_layers, optionally bidirectional)
                  →  Linear(H → 1)  →  Sigmoid
where F is the MFCC feature dim and H is the hidden size (default 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SADGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 5,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bool(bidirectional)

        self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(
        self,
        x: torch.Tensor,                # (B, T, F)
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return frame-level speech *logits* of shape (B, T).

        We return logits, not probabilities, so the loss can use
        `binary_cross_entropy_with_logits` (numerically stable).
        """
        B, T, F = x.shape
        # BatchNorm wants (B, F, T)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            out, _ = self.gru(x)

        logits = self.fc(out).squeeze(-1)  # (B, T)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, lengths))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
