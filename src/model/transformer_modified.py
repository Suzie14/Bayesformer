import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


from src.configs import constants, names
from src.model.attention_modified import (
    FeedForward,
    MultiHeadAttention,
    NormLayer,
    PositionalEmbedding,
)

class Decoder(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            nb_heads=nb_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            mask=True,
        )
        self.norm_layer_1 = NormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = NormLayer(dimension=embedding_dimension)
        self.feed_forward = FeedForward(
            input_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=embedding_dimension,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out_attn = self.attention(q=x, k=x, v=x, mask=mask)
        x = x + self.dropout(out_attn)
        x = self.norm_layer_1(x)

        out_ff = self.feed_forward(x)
        x = x + self.dropout(out_ff)
        x = self.norm_layer_2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        nb_layers: int,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.pos_embedding = PositionalEmbedding(embedding_dimension, dropout)
        self.layers = nn.ModuleList(
            [
                Decoder(
                    nb_heads=nb_heads,
                    embedding_dimension=embedding_dimension,
                    head_size=head_size,
                    hidden_dimension=hidden_dimension,
                    dropout=dropout,
                )
                for _ in range(nb_layers)
            ]
        )
        self.linear = nn.Linear(embedding_dimension, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        T, B = input.shape
        x = self.embedding(input)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        mask = self._generate_square_subsequent_mask(T).to(input.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

        for layer in self.layers:
            x = layer(x, mask=mask)

        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)