"""Transformer class"""

import pandas as pd
import time
import torch
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple

from src.configs import names, constants

from src.model.bayes_attention import (
    MultiHeadAttention,
    NormLayer,
    FeedForward,
    PositionalEncoding,
)


_Decoder = typing.TypeVar(name="_Decoder", bound="Decoder")


class Decoder(torch.nn.Module):
    """
    Class for the Decoder part of the Transformer.

    Methods:
        forward(x): Computes the output of the Decoder.
    """

    def __init__(
        self: _Decoder,
        num_heads: int,
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
            num_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding.
            head_size (int): Size of each single head.
            context_length (int): Lenght of the context.
            hidden_dimension (int): Dimension of the hidden layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            dropout=dropout,
            mask=False,
        )
        self.norm_layer_1 = NormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = NormLayer(dimension=embedding_dimension)
        self.feed_forward = FeedForward(
            input_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=embedding_dimension,
            dropout=dropout,
        )
        self.dropout = torch.nn.Dropout(dropout)

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
        out = self.norm_layer_1(out + self.dropout(input=out))
        out = self.feed_forward(x=out)
        out = self.norm_layer_2(out + self.dropout(input=out))
        return out


_Transformer = typing.TypeVar(name="_Transformer", bound="Transformer")


class Transformer(torch.nn.Module):
    """
    Class for Transformer model.

    Methods:
        forward(input): get predictions of the model.
        train_model(train_dataloader, valid_dataloader): train the model.
    """

    def __init__(self: _Transformer, params: Dict[str, Any]) -> None:
        """
        Initialize class instance.

        Args:
            self (_Transformer): Class instance.
            params (Dict[str, Any]): Parameters of the model.
        """
        super().__init__()
        self.params = params
        if (self.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            self.params[names.DEVICE] = "cuda"
        else:
            self.params[names.DEVICE] = "cpu"
        self.embedding = torch.nn.Embedding(
            self.params[names.SRC_VOCAB_SIZE], self.params[names.EMBEDDING_DIMENSION]
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
            context_length=self.params[names.CONTEXT_LENGTH],
        )
        self.layers = torch.nn.ModuleList(
            [
                Decoder(
                    num_heads=self.params[names.NB_HEADS],
                    embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
                    head_size=self.params[names.HEAD_SIZE],
                    context_length=self.params[names.CONTEXT_LENGTH],
                    hidden_dimension=self.params[names.FEEDFORWARD_DIMENSION],
                    dropout=self.params[names.DROPOUT],
                )
                for _ in range(self.params[names.NB_LAYERS])
            ]
        )
        self.linear = torch.nn.Linear(
            self.params[names.EMBEDDING_DIMENSION],
            self.params[names.TGT_VOCAB_SIZE],
        )
        self.dropout = torch.nn.Dropout(self.params[names.DROPOUT])

    def forward(self: _Transformer, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method to get the output of the Transformer.

        Args:
            self (_Transformer): Class instance.
            input (torch.Tensor): Source input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        src_embedded = self.embedding(self.dropout(input))
        output = self.dropout(
            src_embedded + self.positional_encoding(x=self.dropout(src_embedded))
        )
        for layer in self.layers:
            output = self.dropout(layer(x=output))
        logits = self.linear(output)
        return logits

    def train_model(
        self: _Transformer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> Tuple[List[int], List[int]]:
        """
        Train the model.

        Args:
            self (_Transformer): Class instance.
            train_dataloader (DataLoader): Dataloader for training data.
            valid_dataloader (DataLoader): Dataloader for validation data.

        Returns:
            Tuple[List[int], List[int]]: Train loss history, validation loss history.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params[names.LEARNING_RATE],
            betas=self.params[names.BETAS],
            eps=self.params[names.EPSILON],
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.to(self.params[names.DEVICE])
        train_loss_history = []
        valid_loss_history = []
        start_training = time.time()
        for epoch in range(self.params[names.NB_EPOCHS]):
            # Training
            self.train()
            train_loss = 0.0
            for src, tgt_input, tgt_output in train_dataloader:
                src, tgt_input, tgt_output = (
                    src.to(self.params[names.DEVICE]),
                    tgt_input.to(self.params[names.DEVICE]),
                    tgt_output.to(self.params[names.DEVICE]),
                )
                optimizer.zero_grad()
                logits = self(src, tgt_input)
                B, T, _ = logits.shape
                loss = criterion(
                    logits.view(B * T, self.params[names.TGT_VOCAB_SIZE]),
                    tgt_output.view(B * T),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
                train_loss += loss.item()
            # Validation
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for src, tgt_input, tgt_output in valid_dataloader:
                    src, tgt_input, tgt_output = (
                        src.to(self.params[names.DEVICE]),
                        tgt_input.to(self.params[names.DEVICE]),
                        tgt_output.to(self.params[names.DEVICE]),
                    )
                    logits = self(src, tgt_input)
                    B, T, _ = logits.shape
                    loss = criterion(
                        logits.reshape(B * T, self.params[names.TGT_VOCAB_SIZE]),
                        tgt_output.reshape(B * T),
                    )
                    valid_loss += loss.item()
            ###
            train_loss /= len(train_dataloader)
            valid_loss /= len(valid_dataloader)
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            print(f"Epoch {epoch+1} / {self.params[names.NB_EPOCHS]} -------------")
            print(f"Train loss : {train_loss:.4f}. Valid loss : {valid_loss:.4f}.")
        print(
            f"Trained successfully. It took {time.time() - start_training:.2f} seconds. \n"
        )
        return train_loss_history, valid_loss_history
