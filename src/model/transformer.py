"""Transformer class"""

import time
import typing
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.configs import constants, names
from src.model.attention import (
    FeedForward,
    MultiHeadAttention,
    NormLayer,
    PositionalEncoding,
)

_Decoder = typing.TypeVar(name="_Decoder", bound="Decoder")


class Decoder(nn.Module):
    """
    Class for the Decoder part of the Transformer.

    Methods:
        forward(x): Computes the output of the Decoder.
    """

    def __init__(
        self: _Decoder,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_Decoder): Class instance.
            nb_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding.
            head_size (int): Size of each single head.
            context_length (int): Length of the context.
            hidden_dimension (int): Dimension of the hidden layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            nb_heads=nb_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
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

    def forward(self: _Decoder, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to get the output of the decoder.

        Args:
            self (_Decoder): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.attention(x)
        out = self.norm_layer_1(x + self.dropout(out))
        out = self.feed_forward(out)
        out = self.norm_layer_2(x + self.dropout(out))
        return out


_Transformer = typing.TypeVar(name="_Transformer", bound="Transformer")


class Transformer(nn.Module):
    """
    Class for Transformer model.

    Methods:
        forward(input): get predictions of the model.
        train_model(train_dataloader, valid_dataloader): train the model.
    """

    def __init__(
        self: _Transformer,
        vocab_size: int,
        nb_layers: int,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_Former): Class instance.
            vocab_size (int): Size of the vocabulary.
            nb_layers (int): Number of layers.
            nb_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding.
            head_size (int): Size of the head.
            context_length (int): Length of the context.
            hidden_dimension (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(
            embedding_dimension=embedding_dimension,
            context_length=context_length,
        )
        self.layers = nn.ModuleList(
            [
                Decoder(
                    nb_heads=nb_heads,
                    embedding_dimension=embedding_dimension,
                    head_size=head_size,
                    context_length=context_length,
                    hidden_dimension=hidden_dimension,
                    dropout=dropout,
                )
                for _ in range(nb_layers)
            ]
        )
        self.linear = nn.Linear(
            embedding_dimension,
            vocab_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self: _Transformer, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method to get the output of the Transformer.

        Args:
            self (_Transformer): Class instance.
            input (torch.Tensor): Source input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        out = self.embedding(input)
        out = self.dropout(self.positional_encoding(out))
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
