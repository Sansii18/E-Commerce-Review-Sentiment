"""
Stage 1: LSTM Sentiment Classifier
====================================
4-layer stacked LSTM + single-head attention → binary sentiment (positive/negative).

Architecture
------------
  Embedding (128-dim, learnable)
  → 4 × LSTM (256 hidden units, Dropout 0.3 between layers)
  → Attention pooling (highlights sentiment-driving tokens)
  → FC: 256 → 64 → 1 (sigmoid)

Output
------
  sentiment_prob  : float in [0, 1]   (1 = positive sentiment)
  attention_weights: (seq_len,)        for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


CONFIG = {
    'vocab_size'    : 20_000,
    'embedding_dim' : 128,
    'hidden_dim'    : 256,
    'num_layers'    : 4,
    'dropout'       : 0.3,
    'fc_hidden'     : 64,
    'max_seq_len'   : 200,
    'pad_idx'       : 0,
}


class Attention(nn.Module):
    """Single-head additive attention over LSTM outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn  = nn.Linear(hidden_dim, hidden_dim)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        lstm_out    : torch.Tensor,   # (B, T, H)
        padding_mask: torch.Tensor,   # (B, T)  True where PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context   : (B, H)   — attended context vector
        attn_w    : (B, T)   — attention weights
        """
        scores = self.v(torch.tanh(self.attn(lstm_out))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(padding_mask, float('-inf'))
        attn_w = F.softmax(scores, dim=-1)                             # (B, T)
        # Replace NaN rows (full padding) with uniform attention
        nan_mask = torch.isnan(attn_w).all(dim=-1, keepdim=True)
        attn_w   = torch.where(nan_mask, torch.ones_like(attn_w) / attn_w.size(1), attn_w)
        context  = (attn_w.unsqueeze(-1) * lstm_out).sum(dim=1)        # (B, H)
        return context, attn_w


class SentimentLSTM(nn.Module):
    """
    4-layer stacked LSTM with attention for binary sentiment classification.

    Parameters
    ----------
    vocab_size   : int
    embedding_dim: int
    hidden_dim   : int
    num_layers   : int   (4 by default)
    dropout      : float
    fc_hidden    : int   (intermediate FC dimension)
    pad_idx      : int
    """

    def __init__(
        self,
        vocab_size   : int   = CONFIG['vocab_size'],
        embedding_dim: int   = CONFIG['embedding_dim'],
        hidden_dim   : int   = CONFIG['hidden_dim'],
        num_layers   : int   = CONFIG['num_layers'],
        dropout      : float = CONFIG['dropout'],
        fc_hidden    : int   = CONFIG['fc_hidden'],
        pad_idx      : int   = CONFIG['pad_idx'],
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx    = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size   = embedding_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout,          # between layers
            bidirectional= False,
        )
        self.attention = Attention(hidden_dim)
        self.fc1   = nn.Linear(hidden_dim, fc_hidden)
        self.fc2   = nn.Linear(fc_hidden, 1)
        self.bn    = nn.BatchNorm1d(fc_hidden)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,       # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        prob    : (B, 1)  — sentiment probability (sigmoid)
        attn_w  : (B, T)  — per-token attention weights
        """
        pad_mask = (x == self.pad_idx)                  # (B, T)
        emb      = self.dropout(self.embedding(x))      # (B, T, E)
        out, _   = self.lstm(emb)                       # (B, T, H)
        context, attn_w = self.attention(out, pad_mask) # (B, H), (B, T)
        h   = F.relu(self.bn(self.fc1(self.dropout(context))))
        out = torch.sigmoid(self.fc2(self.dropout(h)))  # (B, 1)
        return out, attn_w

    def save_model(self, path: str) -> None:
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size'   : self.embedding.num_embeddings,
                'embedding_dim': self.embedding.embedding_dim,
                'hidden_dim'   : self.hidden_dim,
                'num_layers'   : self.num_layers,
                'pad_idx'      : self.pad_idx,
            }
        }, path)
        print(f"✅ Sentiment model saved → {path}")

    @classmethod
    def load_model(cls, path: str, device: str = 'cpu') -> 'SentimentLSTM':
        data  = torch.load(path, map_location=device)
        model = cls(**data['config'])
        model.load_state_dict(data['state_dict'])
        model.eval()
        return model