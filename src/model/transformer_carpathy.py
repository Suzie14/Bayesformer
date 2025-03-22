import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# class SelfAttentionHead(nn.Module):
#     def __init__(self, d_model, d_k):
#         super().__init__()
#         self.query = nn.Linear(d_model, d_k)
#         self.key = nn.Linear(d_model, d_k)
#         self.value = nn.Linear(d_model, d_k)
#         self.scale = 1.0 / math.sqrt(d_k)

#     def forward(self, x, mask=None):
#         Q, K, V = self.query(x), self.key(x), self.value(x)
#         attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#         # print(mask.shape, attn_weights.shape)
#         if mask is not None:
#             expected_size = attn_weights.shape[-1]  # 20 dans ce cas
#             mask = F.pad(
#                 mask,
#                 (
#                     0,
#                     expected_size - mask.shape[-1],
#                     0,
#                     expected_size - mask.shape[-2],
#                 ),
#                 value=1,
#             )

#             attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
#         attn_weights = F.softmax(attn_weights, dim=-1)
#         return torch.matmul(attn_weights, V)


class Head(nn.Module):
    def __init__(self, head_input_dim, head_size, head_output_dim):
        super().__init__()
        self.key = nn.Linear(head_input_dim, head_size, bias=False)
        self.query = nn.Linear(head_input_dim, head_size, bias=False)
        self.value = nn.Linear(head_input_dim, head_output_dim, bias=False)
        # Some Pytorch way of defining a matrix without trainable parameters
        self.register_buffer("tril", torch.tril(torch.ones(20, 20)))

        self.head_size = head_size

    def forward(self, x, mask):
        B, T, C = x.shape
        # if training: B = batch_size, else B = 1
        # T = 20
        # I = head_input_dim
        # H = head_size
        # O = head_output_dim

        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, O)
        attention_scores = q @ k.transpose(
            1, 2
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
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
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.heads = nn.ModuleList(
            [Head(d_model, self.d_k, d_model) for _ in range(n_heads)]
        )
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        return self.fc(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = TransformerEncoder(ninp, nhead, nhid, nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)).unsqueeze(0)  # (1, sz, sz)
        return mask

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.input_emb(src) * math.sqrt(self.input_emb.embedding_dim)
        src = self.pos_encoder(src)
        output_enc = self.encoder(src, mask)
        output_dec = self.decoder(output_enc)
        return F.log_softmax(output_dec, dim=-1)
