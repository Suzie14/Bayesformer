"""Class for attention"""

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

_BayesHead = typing.TypeVar("_BayesHead", bound="BayesHead")


class BayesHead(nn.Module):
    """
    A single head of multi-head attention.

    Args:
        head_input_dimension (int): The dimension of the input to the head.
        head_size (int): The size of the head.
        head_output_dimension (int): The dimension of the output of the head.
        context_length (int): The length of the context.
        mask (bool, optional): Whether to apply masking. Defaults to True.

    Methods:
        forward(q, k, v): Computes the attention scores and context vectors.
    """

    def __init__(
        self: _BayesHead,
        head_input_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        dropout: float,
        mask: bool = True,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_BayesHead): Class instance.
            head_input_dimension (int): Dimension of the input.
            head_size (int): Size of the attention head.
            head_output_dimension (int): Dimension of the output.
            context_length (int): Length of the context.
            dropout (float): Dropout rate.
            mask (bool, optional): Whether to use a mask or not. Defaults to True.
        """
        super().__init__()
        self.key = nn.Linear(head_input_dimension, head_size, bias=False)
        self.query = nn.Linear(head_input_dimension, head_size, bias=False)
        self.value = nn.Linear(
            head_input_dimension, head_output_dimension, bias=False
        )
        # Some Pytorch way of defining a matrix without trainable parameters
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length), diagonal=1),
        )  # Not trainable parameters
        self.head_size = head_size
        self.dropout = dropout
        self.mask = mask

    def forward(
        self: _BayesHead,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method to get the output of the model.

        Args:
            self (_BayesHead): Class instance.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output of the model.
        """
        B, T, C = q.shape
        # B = batch_size
        # T = context_length
        # I = head_input_dimension
        # H = head_size
        # O = head_output_dimension
        K = self.key(F.dropout(k, p=self.dropout, training=True))  # (B, T, H)
        Q = self.query(F.dropout(q, p=self.dropout, training=True))  # (B, T, H)
        V = self.value(F.dropout(v, p=self.dropout, training=True))  # (B, T, O)
        attention_scores = Q @ K.transpose(
            1, 2
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        attention_scores = attention_scores * self.head_size**-0.5  # (B, T, T)
        if self.mask:
            masked_attention_scores = attention_scores.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )  # (B, T, T)
        else:
            masked_attention_scores = attention_scores  # (B, T, T)
        attention_weights = torch.softmax(
            masked_attention_scores * self.head_size**-0.5, dim=-1
        )  # (B, T, T)
        context_vectors = (
            attention_weights @ V
        )  # (B, T, T) @ (B, T, O) -> (B, T, O)
        return context_vectors


_BayesMultiHeadAttention = typing.TypeVar(
    "_BayesMultiHeadAttention", bound="BayesMultiHeadAttention"
)


class BayesMultiHeadAttention(nn.Module):
    """
    Multi-head attention.

    Args:
        nb_heads (int): Number of heads.
        embedding_dimension (int): Dimension of the embedding (=input).
        head_size (int): The size of the head.
        head_output_dimension (int): The dimension of the output of the head.
        context_length (int): The length of the context.
        mask (bool, optional): Whether to apply masking. Defaults to True.

    Methods:
        forward(q, k, v): Computes the attention scores and context vectors.
    """

    def __init__(
        self: _BayesMultiHeadAttention,
        nb_heads: int,
        embedding_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        dropout: float,
        mask: bool = True,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_BayesMultiHeadAttention): Class instance.
            nb_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding (=input).
            head_size (int): Size of the attention head.
            head_output_dimension (int): Dimension of the output.
            context_length (int): Length of the context.
            dropout (float): Dropout rate.
            mask (bool, optional): Whether to use a mask or not. Defaults to True.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                BayesHead(
                    head_input_dimension=embedding_dimension,
                    head_size=head_size,
                    head_output_dimension=head_output_dimension,
                    context_length=context_length,
                    dropout=dropout,
                    mask=mask,
                )
                for _ in range(nb_heads)
            ]
        )
        self.proj = nn.Linear(
            nb_heads * head_output_dimension, embedding_dimension
        )
        self.dropout = dropout

    def forward(
        self: _BayesMultiHeadAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Forward method to get the output of the model.

        Args:
            self (_MultiHead): Class instance.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output of the model.
        """
        # q, k, v : (B, T, C)
        # B = batch_size
        # T = context_length
        # C = embedding_dimension
        out = torch.cat(
            [
                F.dropout(h(q=q, k=k, v=v), p=self.dropout, training=True)
                for h in self.heads
            ],
            dim=-1,
        )
        out = self.proj(out)
        return out  # Output : (B, T, C)


_NormLayer = typing.TypeVar("_NormLayer", bound="NormLayer")


class NormLayer(nn.Module):
    """
    Normalization layer.

    Args:
        dimension (int): The dimension of the input and output tensors.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Methods:
        forward(x): Applies layer normalization to the input tensor.
    """

    def __init__(self: _NormLayer, dimension: int, eps: float = 1e-6) -> None:
        """
        Initialize class instance.

        Args:
            self (_NormLayer): Class instance.
            dimension (int): Dimension of the input tensor.
            eps (float, optional): To avoid dividing by 0. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps  # To avoid dividing by 0
        self.gamma = nn.Parameter(torch.ones(dimension))  # Parameter to train
        self.beta = nn.Parameter(torch.zeros(dimension))  # Parameter to train

    def forward(self: _NormLayer, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to get the output of the layer.

        Args:
            self (_NormLayer): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # x : (B, T, C)
        # B = batch_size
        # T = length
        # C = dimension
        mean = x.mean(dim=-1, keepdim=True)  # Mean : (B, T, C)
        std = torch.sqrt(
            torch.var(x, dim=-1, unbiased=False, keepdim=True)
        )  # Standard deviation : (B, T, C)
        x_normalized = (x - mean) / (
            std + self.eps
        )  # Normalized tensor : (B, T, C)
        return self.gamma * x_normalized + self.beta  # Output : (B, T, C)


_BayesFeedForward = typing.TypeVar(
    "_BayesFeedForward", bound="BayesFeedForward"
)


class BayesFeedForward(nn.Module):
    """
    Feed-forward layer.

    Args:
        input_dimension (int): The dimension of the input tensor.
        hidden_dimension (int): The dimension of the hidden layer.
        output_dimension (int): The dimension of the output tensor.

    Methods:
        forward(x): Applies a feed-forward network to the input tensor.
    """

    def __init__(
        self: _BayesFeedForward,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float,
    ):
        """
        Initialize the feed-forward network.

        Args:
            self (_BayesFeedForward): Class instance.
            input_dimension (int): Dimension of the input tensor.
            hidden_dimension (int): Hidden dimension of the network.
            output_dimension (int): Dimension of the output tensor.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, output_dimension),
        )
        self.dropout = dropout

    def forward(self: _BayesFeedForward, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the output of the network.

        Args:
            self (_BayesFeedForward): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x : (B, T, I)
        # B = batch_size
        # T = context_length
        # I = input_dimension
        # O = output_dimension
        return self.net(
            F.dropout(x, p=self.dropout, training=True)
        )  # Output : (B, T, O)


_PositionalEncoding = typing.TypeVar(
    "_PositionalEncoding", bound="PositionalEncoding"
)


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding layer.

    Args:
        embedding_dimension (int): The dimension of the input and output tensors.
        context_length (int): The length of the context.

    Methods:
        forward(x): Applies positional encoding to the input tensor.
    """

    def __init__(
        self: _PositionalEncoding,
        embedding_dimension: int,
        context_length: int,
        dropout: float,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_PositionalEncoding): Class instance.
            embedding_dimension (int): Dimension of the embedding.
            context_length (int): Length of the context.
            dropout (float): Dropout rate.
        """
        super().__init__()
        pe = torch.zeros(context_length, embedding_dimension)
        position = torch.arange(
            start=0, end=context_length, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float()
            * -(math.log(10000.0) / embedding_dimension)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = dropout

    def forward(self: _PositionalEncoding, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the output of the embedding.

        Args:
            self (_PositionalEncoding): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor.
        """
        return x + F.dropout(
            self.pe[:, : x.size(1)], p=self.dropout, training=True
        )
