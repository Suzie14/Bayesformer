import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# class MultiHeadAttention(nn.Module):
#     """
#     Multi-head self-attention, format (T, B, E).
#     mask shape: (T, T) if needed for causal.
#     """

#     def __init__(self, d_model, n_heads, d_ff, causal=True):
#         """
#         Args:
#             d_model (int): embedding dim
#             n_heads (int): number of attention heads
#             d_ff (int): feed-forward dimension for each head
#             causal (bool): whether we use a causal mask
#         """
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads  # dimension per head
#         self.causal = causal

#         # Query, Key, Value
#         self.W_q = nn.Linear(d_model, d_model, bias=False)
#         self.W_k = nn.Linear(d_model, d_model, bias=False)
#         self.W_v = nn.Linear(d_model, d_model, bias=False)
#         # Final projection
#         self.W_out = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, x, mask=None):
#         """
#         x: (T, B, E)
#         mask: (T, T) or None
#         Return: (T, B, E)
#         """
#         T, B, E = x.shape
#         # 1) compute Q,K,V
#         Q = self.W_q(x)
#         K = self.W_k(x)
#         V = self.W_v(x)

#         # 2) split heads
#         # => (T, B, nHeads, d_head) => (T, B*nHeads, d_head)
#         Q = Q.view(T, B, self.n_heads, self.d_head).transpose(
#             1, 2
#         )  # (T, nHeads, B, d_head)
#         K = K.view(T, B, self.n_heads, self.d_head).transpose(1, 2)
#         V = V.view(T, B, self.n_heads, self.d_head).transpose(1, 2)
#         # => (T, nHeads*B, d_head) en combinant nHeads et B
#         Q = Q.contiguous().view(T, -1, self.d_head)  # (T, nHeads*B, d_head)
#         K = K.contiguous().view(T, -1, self.d_head)
#         V = V.contiguous().view(T, -1, self.d_head)

#         QT = Q.transpose(0, 1)  # => (nHB, T, d_head)
#         KT = K.transpose(0, 1).transpose(1, 2)  # => (nHB, d_head, T)
#         scores = QT.bmm(KT)  # => (nHB, T, T)
#         scores = scores / math.sqrt(self.d_head)
#         # Optional causal mask
#         if self.causal and mask is not None:
#             scores = scores + mask  # mask is negative inf on upper triangle
#         attn_weights = F.softmax(scores, dim=-1)  # (nHB, T, T)
#         # multiply by V
#         VT = V.transpose(0, 1)  # => (nHB, T, d_head)
#         out = attn_weights.bmm(VT)  # => (nHB, T, d_head)
#         # out => transpose back => (T, nHB, d_head)
#         out = out.transpose(0, 1).contiguous()  # => (T, nHB, d_head)
#         # merge heads => (T, B, E)
#         out = out.view(T, self.n_heads, B, self.d_head).transpose(
#             1, 2
#         )  # => (T, B, nHeads, d_head)
#         out = out.contiguous().view(T, B, E)
#         # final projection
#         out = self.W_out(out)
#         return out


class Head(nn.Module):
    def __init__(
        self, d_model, d_ff, causal=True, bayes=False, bayes_dropout=None
    ):
        super().__init__()
        self.d_head = d_ff
        self.causal = causal
        self.bayes = bayes
        self.bayes_dropout = bayes_dropout

        self.W_q = nn.Linear(d_model, d_ff, bias=False)
        self.W_k = nn.Linear(d_model, d_ff, bias=False)
        self.W_v = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x, mask=None):
        # T, B, _ = x.shape
        if self.bayes:
            # print("bayes")
            Q = self.W_q(F.dropout(x, p=self.bayes_dropout, training=True))
            K = self.W_k(F.dropout(x, p=self.bayes_dropout, training=True))
            V = self.W_v(F.dropout(x, p=self.bayes_dropout, training=True))
        else:
            Q = self.W_q(x)  # (T, B, d_head)
            K = self.W_k(x)
            V = self.W_v(x)

        scores = (
            Q.transpose(0, 1) @ K.transpose(0, 1).transpose(1, 2)
        ) / math.sqrt(self.d_head)
        if self.causal and mask is not None:
            scores = scores + mask  # mask is negative inf on upper triangle
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V.transpose(0, 1)
        return out.transpose(0, 1)  # (T, B, d_head)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        causal=True,
        bayes_dropout=None,
        bayes=False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.bayes = bayes
        self.bayes_dropout = bayes_dropout

        self.heads = nn.ModuleList(
            [Head(d_model, self.d_head, causal) for _ in range(n_heads)]
        )
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        if self.bayes:
            # print("bayes")
            heads_out = torch.cat(
                [
                    F.dropout(
                        head(x, mask), p=self.bayes_dropout, training=True
                    )
                    for head in self.heads
                ],
                dim=-1,
            )
        else:
            heads_out = torch.cat(
                [head(x, mask) for head in self.heads], dim=-1
            )
        return self.W_out(heads_out)


#  F.dropout(h(q=q, k=k, v=v), p=self.dropout, training=True)
class FeedForward(nn.Module):
    """
    Simple feed-forward network.
    In PyTorch Transformer, dim_feedforward=some factor * d_model
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        # Bayesian
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        out = self.net(x)
        if self.dropout is not None:
            return F.dropout(out, p=self.dropout, training=True)
        else:
            return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder block (self-attn + feed-forward + residuals).
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout=0.1,
        causal=True,
        bayes_dropout=None,
        bayes=False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, d_ff=dim_feedforward, causal=causal
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, bayes_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.bayes = bayes
        self.bayes_dropout = bayes_dropout

    def forward(self, x, mask=None):
        # 1) self-attn
        if self.bayes:
            out = self.self_attn(x, mask=mask)  # (T, B, d_model)
            out = self.norm1(
                x + F.dropout(out, p=self.bayes_dropout, training=True)
            )
            out = self.ff(out)  # (T, B, d_model)
            out = self.norm2(
                x + F.dropout(out, p=self.bayes_dropout, training=True)
            )
        else:
            out = self.self_attn(x, mask=mask)  # (T, B, d_model)
            out = self.norm1(x + self.dropout(out))
            out = self.ff(out)  # (T, B, d_model)
            out = self.norm2(x + self.dropout(out))
            return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # => (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: (T, B, d_model)
        """
        T, B, E = x.shape
        # add positional encoding up to T steps
        x = x + self.pe[:T, :]
        return self.dropout(x)


def generate_subsequent_mask(sz):
    """
    Return a (sz, sz) mask:
    lower-triangular => 0
    upper-triangular => -inf
    so that we can't attend to future tokens
    """
    mask = torch.full((sz, sz), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask


class MyTransformer(nn.Module):
    """
    Full 'decoder-only' Transformer in format (T, B, E).
    """

    def __init__(
        self,
        ntoken,
        d_model,
        n_heads,
        dim_feedforward,
        nlayers,
        dropout=0.1,
        bayes_dropout=None,
        bayes=False,
    ):
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout=dropout,
                    causal=True,
                )
                for _ in range(nlayers)
            ]
        )
        self.linear_out = nn.Linear(d_model, ntoken)
        self._init_weights()
        self.bayes = bayes
        self.bayes_dropout = bayes_dropout

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, src):
        """
        src shape: (T, B), containing token IDs
        Returns: (T, B, ntoken) => log_probs
        """
        # 1) Embed => (T, B, d_model)
        if self.bayes:
            # print("bayes")
            x = F.dropout(
                self.embedding(src), p=self.bayes_dropout, training=True
            ) * math.sqrt(self.d_model)
        else:
            x = self.embedding(src) * math.sqrt(self.d_model)
        # 2) Positional encoding => (T, B, d_model)
        x = self.pos_enc(x)
        # 3) Build causal mask
        T, B = src.shape
        mask = generate_subsequent_mask(T).to(src.device)  # (T, T)
        # 4) Pass through N layers
        for layer in self.layers:
            if self.bayes:
                # print("bayes")
                x = F.dropout(
                    layer(x, mask=mask), p=self.bayes_dropout, training=True
                )
            else:
                x = layer(x, mask=mask)
        # 5) Final projection => logits => log_softmax
        logits = self.linear_out(x)  # (T, B, ntoken)
        return F.log_softmax(logits, dim=-1)
