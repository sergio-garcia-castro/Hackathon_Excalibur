"""
Model definitions for learning voice embeddings.

Hierarchical Transformer embedding model:
  - Intra-session/day attention over chunks -> session vector
  - Inter-session attention over session vectors -> final embedding

Designed for LOPO training with a forced linear head.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

## DayEmbeddingsModel (Model to generate day embeddings)
class ChunkMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: [N, K, F]
        return self.net(x)  # [N, K, D]

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, mask):
        """
        h:    [N, K, D]
        mask: [N, K] bool, True = valid
        """
        scores = self.score(h).squeeze(-1)          # [N, K]
        scores = scores.masked_fill(~mask, -1e9)    # ignore padding
        w = torch.softmax(scores, dim=1)            # [N, K]
        pooled = torch.sum(h * w.unsqueeze(-1), dim=1)  # [N, D]
        return pooled, w

class DayEmbeddingModel(nn.Module):
    def __init__(self, in_dim, chunk_hidden, day_dim, dropout=0.1):
        super().__init__()
        self.chunk_encoder = ChunkMLP(in_dim, chunk_hidden, day_dim, dropout)
        self.pool = AttnPool(day_dim)

    def forward(self, x, chunk_mask):
        """
        x: [N, K, F]        where N = B*T (flattened)
        chunk_mask: [N, K]
        returns: day_emb [N, D]
        """
        h = self.chunk_encoder(x)                 # [N, K, D]
        day_emb, attn_w = self.pool(h, chunk_mask)
        return day_emb  # (optionally also return attn_w)

## PositionalEmbeddings model (Add poisitional embeddigs based on position)
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=50):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, T, device):
        pos = torch.arange(T, device=device)      # [T]
        return self.emb(pos)                      # [T, D]

## DeltaDaysEmbeddings model (encode temporal difference based on the difference between consecutive samples)
class DeltaDaysEmbedding(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, delta_days):
        """
        delta_days: [B,T] float
        returns:    [B,T,D]
        """
        return self.mlp(delta_days.unsqueeze(-1))

## SessionModel (Transformer implementation to get patient embeddings)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb, n_heads, dropout=0.1):
        super().__init__()

        assert d_emb % n_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.head_dim = d_emb // n_heads

        self.q = nn.Linear(d_emb, d_emb, bias=False)
        self.k = nn.Linear(d_emb, d_emb, bias=False)
        self.v = nn.Linear(d_emb, d_emb, bias=False)
        # self.c_attn = nn.Linear(d_emb, 3*d_emb, bias=False) then use .split to obtain q,k,v
        self.W_out = nn.Linear(d_emb, d_emb, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, att_mask=None):
        batch_size, seq_len, _ = x.size() # (B, S, D)

        # 1) Project
        Q = self.q(x)  # (B, S, D)
        K = self.k(x)  # (B, S, D)
        V = self.v(x)  # (B, S, D)
        
        # If used s single linear module (c_attn) to obtain q,k,v 
        # qkv = self.c_attn(x)
        # q, k, v = qkv.split(self.n_embd, dim=2)
        # Q = q(x)  # (B, S, D)
        # K = k(x)  # (B, S, D)
        # V = v(x)  # (B, S, D)

        # 2) Split heads: (B, H, S, Dh)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product attention scores: (B, H, S, S)
        att_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4) Attention mask (optional) (broadcast to (B, H, S, S)))
        if att_mask is not None:
            # convert 0/1 -> bool if needed
            if att_mask.dtype != torch.bool:
                att_mask = att_mask != 0

            if att_mask.dim() == 2:
                # (S, S) -> (1, 1, S, S)
                att_mask = att_mask.unsqueeze(0).unsqueeze(0)
            elif att_mask.dim() == 3:
                # (B, S, S) -> (B, 1, S, S)
                att_mask = att_mask.unsqueeze(1)
            else:
                raise ValueError("att_mask must have shape (S,S) or (B,S,S)")
            
            att_scores = att_scores.masked_fill(~att_mask, -1e9)

        # 5) Softmax
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights)

        # 6) Weighted sum: (B, H, S, Dh)
        head_out = att_weights @ V

        # 7) Concatenate heads: (B, S, D)
        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_emb)

        # 8) Output projection
        out = self.W_out(head_out)

        return out

class FeedForwardNet(nn.Module):
    def __init__(self, d_emb, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_emb, d_ff)
        self.fc2 = nn.Linear(d_ff, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        if self.activation == "relu":
            x = F.relu(x)
        else:
            x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SessionTransformer(nn.Module):
    """
    Input:
      x:        (B, T, D)
      day_mask: (B, T) bool, True for valid tokens

    Output:
      z: (B, D) pooled embedding per patient
    """
    def __init__(self, d_emb, n_heads=4, n_layers=2, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_emb

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.RMSNorm(d_emb),
                "attn":  MultiHeadAttention(d_emb, n_heads, dropout=dropout),
                "norm2": nn.RMSNorm(d_emb),
                "ffn":   FeedForwardNet(d_emb, d_ff, dropout=dropout),
                "drop":  nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])

        self.norm_out = nn.RMSNorm(d_emb)

    def forward(self, x, day_mask):
        B, T, D = x.shape

        # padding mask (valid-to-valid)
        att_mask = day_mask[:, None, :] & day_mask[:, :, None]   # (B, T, T)

        # causal mask (lower triangular)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)

        # combine: allow only (valid keys/queries) AND (past/self)
        att_mask = att_mask & causal[None, :, :]  # (B, T, T)

        for layer in self.layers:
            h = layer["norm1"](x)
            h = layer["attn"](h, att_mask=att_mask)
            x = x + layer["drop"](h)

            h = layer["norm2"](x)
            h = layer["ffn"](h)
            x = x + layer["drop"](h)

        x = self.norm_out(x)  # (B, T, D)
        return x

    
## LinearClassifierHead    
class LinearClassifierHead(nn.Module):
    """
    Simple linear classifier head.

    This is intentionally kept simple to force the embedding model to learn
    rich, discriminative representations. A complex classifier would defeat
    the purpose of learning good embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input embeddings.
    num_classes : int, default=2
        Number of output classes.
    """

    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super(LinearClassifierHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (batch_size, embedding_dim).

        Returns
        -------
        logits : torch.Tensor
            Class logits of shape (batch_size, num_classes).
        """
        return self.linear(embeddings)