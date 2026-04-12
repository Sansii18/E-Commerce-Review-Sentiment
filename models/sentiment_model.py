"""
Stage 1: LSTM-based Sentiment Classifier with Attention

Corresponds to: Practical 8 (LSTM) + Practical 5 (Regularization)

Architecture:
- Embedding layer (vocab_size × 128)
- 4 stacked LSTM layers (hidden=256, bidirectional=False)
- Dropout between layers (0.3)
- Single-head Attention mechanism
- Fully connected layers (256 → 64 → 1)
- Sigmoid output for binary classification

Key feature: Attention mechanism provides interpretability by
highlighting tokens most responsible for sentiment prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class Attention(nn.Module):
    """
    Single-head attention mechanism for sequence-to-scalar prediction.
    
    Learns which tokens in a sequence are most important for
    sentiment classification. Returns both context vector and
    attention weights for explainability.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of LSTM hidden state
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights over sequence.
        
        Args:
            lstm_output: Tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            context: Weighted sum of LSTM outputs (batch_size, hidden_dim)
            attention_weights: Normalized weights (batch_size, seq_len)
        """
        # Compute attention scores for each token
        scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Normalize with softmax
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Compute weighted sum across sequence
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_output  # (batch_size, seq_len, hidden_dim)
        )  # (batch_size, 1, hidden_dim)
        
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        return context, attention_weights


class SentimentLSTM(nn.Module):
    """
    4-layer LSTM with attention for binary sentiment classification.
    
    Key design decisions:
    - 4 LSTM layers: ablation during training showed diminishing returns beyond 4
    - Bidirectional=False: unidirectional LSTM better captures temporal flow
      of sentiment expression in reviews
    - Dropout 0.3: empirically tuned on validation set (see train_sentiment.py)
    - Attention: enables interpretability by revealing which words drive sentiment
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension (128)
            hidden_dim: LSTM hidden state dimension (256)
            num_layers: Number of stacked LSTM layers (4)
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Don't learn embeddings for <PAD> token
        )
        
        # 4-layer LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_dim)
        
        # Classification head
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        token_ids: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for sentiment classification.
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) with token indices
            return_attention: If True, return attention weights
            
        Returns:
            prediction: Tensor of shape (batch_size,) with probabilities 0-1
            attention_weights: (batch_size, seq_len) if return_attention=True
        """
        # Embed tokens
        embedded = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)  # lstm_output: (batch, seq_len, hidden)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_output)
        
        # Classification head
        x = self.relu(self.fc1(context))
        x = self.dropout(x) if hasattr(self, 'dropout') else x
        logits = self.fc2(x)
        prediction = self.sigmoid(logits).squeeze(-1)
        
        if return_attention:
            return prediction, attention_weights
        else:
            return prediction, None
    
    def save_model(self, filepath: str, config: Dict, best_val_accuracy: float = None) -> None:
        """
        Save model, vocabulary, and configuration.
        
        Args:
            filepath: Path to save model
            config: Configuration dict with hyperparameters
            best_val_accuracy: Best validation accuracy achieved
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'config': config,
            'best_val_accuracy': best_val_accuracy
        }
        
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_model(filepath: str, device: str = 'cpu') -> 'SentimentLSTM':
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to saved checkpoint
            device: Device to load model on
            
        Returns:
            SentimentLSTM model with loaded weights
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        model = SentimentLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            bidirectional=checkpoint['bidirectional']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model
