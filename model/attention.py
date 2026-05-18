"""Các module Attention cho mô hình Transformer.

Module này chứa cài đặt attention theo scaled dot-product,
projection cho Q/K/V, và Multi-Head Attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Tính toán scaled dot-product attention."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    """Module Multi-Head Attention.

    Lớp này chiếu đầu vào thành Q/K/V, chia thành nhiều head, áp dụng
    scaled dot-product attention, rồi ghép lại.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)
        self.out_lin = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout)

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)  # [..., num_heads, head_dim]
        return x.transpose(-3, -2)  # move num_heads before seq_len

    def _combine_heads(self, x):
        x = x.transpose(-3, -2)
        new_shape = x.size()[:-2] + (self.embed_dim,)
        return x.contiguous().view(*new_shape)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        x, attn = self.attn(q, k, v, mask=mask)
        x = self._combine_heads(x)
        return self.out_lin(x)
