import torch.nn as nn
import torch
import math

#TODO1
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale_factor, dropout_prob):
        super().__init__()
        self.scale = scale_factor
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, heads=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        self.to_queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.unify_heads = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn = ScaledDotProductAttention(scale_factor=self.head_dim ** 0.5, dropout_prob=dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project and split into heads
        q = self.to_queries(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_keys(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_values(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        attended = self.attn(q, k, v)

        # Concatenate heads
        out = attended.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.unify_heads(out)

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
      
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    