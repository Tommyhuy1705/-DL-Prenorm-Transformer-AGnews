"""Transformer chuẩn (Post-Norm).

LayerNorm được áp dụng sau kết nối residual (post-norm).
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PostNormEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x


class TransformerPostNorm(nn.Module):
    """Bộ mã hóa Transformer tối giản cho phân loại (biến thể post-norm)."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, heads: int = 8, d_ff: int = 512, n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([PostNormEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, src, mask=None):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask=mask)
        # x: (batch, seq_len, d_model) -> pool over seq_len
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)