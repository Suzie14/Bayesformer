import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention, format (T, B, E).
    mask shape: (T, T) if needed for causal.
    """

    def __init__(self, d_model, n_heads, d_ff, causal=True):
        """
        Args:
            d_model (int): embedding dim
            n_heads (int): number of attention heads
            d_ff (int): feed-forward dimension for each head
            causal (bool): whether we use a causal mask
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # dimension per head
        self.causal = causal

        # Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # Final projection
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: (T, B, E)
        mask: (T, T) or None
        Return: (T, B, E)
        """
        T, B, E = x.shape
        # 1) compute Q,K,V
        Q = self.W_q(x)  # (T, B, E)
        K = self.W_k(x)  # (T, B, E)
        V = self.W_v(x)  # (T, B, E)

        # 2) split heads
        # => (T, B, nHeads, d_head) => (T, B*nHeads, d_head)
        Q = Q.view(T, B, self.n_heads, self.d_head).transpose(1, 2)  # (T, nHeads, B, d_head)
        K = K.view(T, B, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(T, B, self.n_heads, self.d_head).transpose(1, 2)
        # => (T, nHeads*B, d_head) en combinant nHeads et B
        Q = Q.contiguous().view(T, -1, self.d_head)  # (T, nHeads*B, d_head)
        K = K.contiguous().view(T, -1, self.d_head)
        V = V.contiguous().view(T, -1, self.d_head)

        # 3) attention : A = softmax( Q.K^T / sqrt(d_head) ), output = A.V
        #   Q: (T, nHB, d_head)
        #   K: (T, nHB, d_head)
        #   want: (nHB, T, T) after Q*K => do batch dim on the 2nd axis
        # let's do: Q^T => (nHB, T, d_head), K => (T, nHB, d_head)
        # we want scores shape => (nHB, T, T).

        QT = Q.transpose(0, 1)  # => (nHB, T, d_head)
        KT = K.transpose(0, 1).transpose(1, 2)  # => (nHB, d_head, T)
        scores = QT.bmm(KT)  # => (nHB, T, T)
        scores = scores / math.sqrt(self.d_head)

        # Optional causal mask
        if self.causal and mask is not None:
            scores = scores + mask  # mask is negative inf on upper triangle

        attn_weights = F.softmax(scores, dim=-1)  # (nHB, T, T)

        # multiply by V
        VT = V.transpose(0, 1)  # => (nHB, T, d_head)
        out = attn_weights.bmm(VT)  # => (nHB, T, d_head)
        # out => transpose back => (T, nHB, d_head)
        out = out.transpose(0, 1).contiguous()  # => (T, nHB, d_head)

        # merge heads => (T, B, E)
        out = out.view(T, self.n_heads, B, self.d_head).transpose(1, 2)  # => (T, B, nHeads, d_head)
        out = out.contiguous().view(T, B, E)

        # final projection
        out = self.W_out(out)
        return out
    
class FeedForward(nn.Module):
    """
    Simple feed-forward network. 
    In PyTorch Transformer, dim_feedforward=some factor * d_model
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder block (self-attn + feed-forward + residuals).
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, causal=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_ff=dim_feedforward, causal=causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape => (T, B, d_model)
        # 1) self-attn
        attn_out = self.self_attn(x, mask=mask)  # (T, B, d_model)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # 2) feed-forward
        ff_out = self.ff(x)  # (T, B, d_model)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
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
        x = x + self.pe[:T, 🙂
        return self.dropout(x)
    

def generate_subsequent_mask(sz):
    """
    Return a (sz, sz) mask:
    lower-triangular => 0
    upper-triangular => -inf
    so that we can't attend to future tokens
    """
    mask = torch.full((sz, sz), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    return mask

class MyTransformer(nn.Module):
    """
    Full 'decoder-only' Transformer in format (T, B, E).
    """

    def __init__(self, ntoken, d_model, n_heads, dim_feedforward, nlayers, dropout=0.1):
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, 
                n_heads, 
                dim_feedforward, 
                dropout=dropout,
                causal=True
            )
            for _ in range(nlayers)
        ])
        self.linear_out = nn.Linear(d_model, ntoken)
        self._init_weights()

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
        x = self.embedding(src) * math.sqrt(self.d_model)

        # 2) Positional encoding => (T, B, d_model)
        x = self.pos_enc(x)

        # 3) Build causal mask
        T, B = src.shape
        mask = generate_subsequent_mask(T).to(src.device)  # (T, T)

        # 4) Pass through N layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        # 5) Final projection => logits => log_softmax
        logits = self.linear_out(x)  # (T, B, ntoken)
        return F.log_softmax(logits, dim=-1)


import torch.optim as optim

# Hyperparams
ntoken = VOCAB_SIZE  # taille du vocab
d_model = 64
nheads = 8
dim_ff = 128
nlayers = 2
dropout = 0.1

device="cpu"
transformer = MyTransformer(ntoken, d_model, nheads, dim_ff, nlayers, dropout).to(device)