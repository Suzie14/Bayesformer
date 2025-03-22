import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

_Head = typing.TypeVar("_Head", bound="Head")


class Head(nn.Module):
    def __init__(
        self: _Head,
        head_input_dimension: int,
        head_size: int,
        head_output_dimension: int,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(head_input_dimension, head_size, bias=False)
        self.query = nn.Linear(head_input_dimension, head_size, bias=False)
        self.value = nn.Linear(head_input_dimension, head_output_dimension, bias=False)
        self.head_size = head_size
        self.mask = mask

    def forward(
        self: _Head, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        T, B, C = q.shape
        K = self.key(k)
        Q = self.query(q)
        V = self.value(v)

        Qp = Q.permute(1, 0, 2)  # (B, T, head_size)
        Kp = K.permute(1, 2, 0)  # (B, head_size, T)
        attention_scores = Qp @ Kp
        attention_scores = attention_scores * self.head_size ** -0.5

        if self.mask and mask is not None:
            attention_scores = attention_scores + mask  # mask shape: (T, T) broadcasted to (B, T, T)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        Vp = V.permute(1, 0, 2)
        context_vectors = attention_weights @ Vp
        context_vectors = context_vectors.permute(1, 0, 2)
        return context_vectors


_MultiHeadAttention = typing.TypeVar("_MultiHeadAttention", bound="MultiHeadAttention")


class MultiHeadAttention(nn.Module):
    def __init__(
        self: _MultiHeadAttention,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        head_output_dimension: int,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_input_dimension=embedding_dimension,
                    head_size=head_size,
                    head_output_dimension=head_output_dimension,
                    mask=mask,
                )
                for _ in range(nb_heads)
            ]
        )
        self.proj = nn.Linear(nb_heads * head_output_dimension, embedding_dimension)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class NormLayer(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


class FeedForward(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, I = x.shape
        out = self.net(x.view(T * B, I))
        return out.view(T, B, -1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T, B, C = x.shape
        x = x + self.pe[:T, :]
        return self.dropout(x)

