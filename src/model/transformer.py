"""Transformer class"""

import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple

from src.configs import names, constants

from src.model.attention import (
    MultiHeadAttention,
    NormLayer,
    FeedForward,
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
        out = self.attention(q=x, k=x, v=x)
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

    # def train_model(
    #     self: _Transformer,
    #     train_dataloader: DataLoader,
    #     valid_dataloader: DataLoader,
    # ) -> Tuple[List[int], List[int]]:
    #     """
    #     Train the model.

    #     Args:
    #         self (_Transformer): Class instance.
    #         train_dataloader (DataLoader): Dataloader for training data.
    #         valid_dataloader (DataLoader): Dataloader for validation data.

    #     Returns:
    #         Tuple[List[int], List[int]]: Train loss history, validation loss history.
    #     """
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.params[names.LEARNING_RATE],
    #         betas=self.params[names.BETAS],
    #         eps=self.params[names.EPSILON],
    #     )
    #     criterion = nn.CrossEntropyLoss(ignore_index=0)
    #     self.to(self.params[names.DEVICE])
    #     train_loss_history = []
    #     valid_loss_history = []
    #     start_training = time.time()
    #     for epoch in range(self.params[names.NB_EPOCHS]):
    #         # Training
    #         self.train()
    #         train_loss = 0.0
    #         for src, tgt_input, tgt_output in train_dataloader:
    #             src, tgt_input, tgt_output = (
    #                 src.to(self.params[names.DEVICE]),
    #                 tgt_input.to(self.params[names.DEVICE]),
    #                 tgt_output.to(self.params[names.DEVICE]),
    #             )
    #             optimizer.zero_grad()
    #             logits = self(src, tgt_input)
    #             B, T, _ = logits.shape
    #             loss = criterion(
    #                 logits.view(B * T, self.params[names.TGT_VOCAB_SIZE]),
    #                 tgt_output.view(B * T),
    #             )
    #             loss.backward()
    #             nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
    #             optimizer.step()
    #             train_loss += loss.item()
    #         # Validation
    #         self.eval()
    #         valid_loss = 0.0
    #         with torch.no_grad():
    #             for src, tgt_input, tgt_output in valid_dataloader:
    #                 src, tgt_input, tgt_output = (
    #                     src.to(self.params[names.DEVICE]),
    #                     tgt_input.to(self.params[names.DEVICE]),
    #                     tgt_output.to(self.params[names.DEVICE]),
    #                 )
    #                 logits = self(src, tgt_input)
    #                 B, T, _ = logits.shape
    #                 loss = criterion(
    #                     logits.reshape(B * T, self.params[names.TGT_VOCAB_SIZE]),
    #                     tgt_output.reshape(B * T),
    #                 )
    #                 valid_loss += loss.item()
    #         ###
    #         train_loss /= len(train_dataloader)
    #         valid_loss /= len(valid_dataloader)
    #         train_loss_history.append(train_loss)
    #         valid_loss_history.append(valid_loss)
    #         print(f"Epoch {epoch+1} / {self.params[names.NB_EPOCHS]} -------------")
    #         print(f"Train loss : {train_loss:.4f}. Valid loss : {valid_loss:.4f}.")
    #     print(
    #         f"Trained successfully. It took {time.time() - start_training:.2f} seconds. \n"
    #     )
    #     return train_loss_history, valid_loss_history
