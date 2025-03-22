import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(
        self,
        head_input_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(head_input_dimension, head_size, bias=False)
        self.query = nn.Linear(head_input_dimension, head_size, bias=False)
        self.value = nn.Linear(
            head_input_dimension, head_output_dimension, bias=False
        )
        # Some Pytorch way of defining a matrix without trainable parameters
        self.context_length = context_length
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length), diagonal=1),
        )  # Not trainable parameters
        self.head_size = head_size
        self.mask = mask

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, T, _ = x.shape
        K = self.key(x)  # (B, T, H)
        Q = self.query(x)  # (B, T, H)
        V = self.value(x)  # (B, T, O)
        attention_scores = Q @ K.transpose(
            1, 2
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        attention_scores = attention_scores * self.head_size**-0.5  # (B, T, T)
        
        masked_attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        attention_weights = torch.softmax(
            masked_attention_scores * self.head_size**-0.5, dim=-1
        )  # (B, T, T)
        context_vectors = (
            attention_weights @ v
        )  # (B, T, T) @ (B, T, O) -> (B, T, O)
        return context_vectors







class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_head, head_size, head_output_dim):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_input_dim=n_embed,
                    head_size=head_size,
                    head_output_dim=head_output_dim,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_head * head_output_dim, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_head, head_size, head_output_dim, n_embed2):
        super().__init__()
        self.self_attention_heads = MultiHeadAttention(
            n_head=n_head, head_size=head_size, head_output_dim=head_output_dim
        )
        self.ffwd = FeedFoward(n_embed2=n_embed2, n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # here are skip connections, also called residual connections
        # they help training deep neural networks by adding a pathway to the input
        x = x + self.self_attention_heads(x)

        # normalization layer; recent implementations put them before self attention heads!
        x = self.ln1(x)

        # and again skip connections:
        x = x + self.ffwd(x)

        # and again normalization layer
        x = self.ln2(x)

        return x


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed2, n_embed=n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed2),
            nn.Tanh(),
            nn.Linear(n_embed2, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class LanguageModel(nn.Module):
    def __init__(
        self,
        n_token,
        n_embed,
        context_length,
        n_head,
        head_size,
        head_output_dim,
        n_embed2,
        n_layer,
    ):
        super().__init__()
        self.
        self.token_embedding_table = nn.Embedding(n_token, n_embed)
        self.position_embedding_table = nn.Embedding(context_length, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(n_head, head_size, head_output_dim, n_embed2)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, n_token)

    def forward(self, idx, y=None):
        B, T = idx.shape
        # I = head_input_dim = head_output_dim = n_embed

        tok_emb = self.token_embedding_table(idx)  # (B, T, I)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T, I)
        x = tok_emb + pos_emb  # (B, T, I)
        x = self.blocks(x)  # (B, T, I)
        x = self.ln_f(x)  # (B, T, I)
        logits = self.lm_head(x)  # (B, T, n_token)

        if y is None:
            loss = None
        else:
            B, T, _ = logits.shape
            logits = logits.view(B * T, n_token)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)

        return logits, loss
