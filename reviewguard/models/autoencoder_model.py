"""
Stage 2: LSTM Autoencoder for Fake Review Detection

Corresponds to: Practical 9 (Autoencoders) + Practical 6 (Anomaly Detection)

Architecture:
- Encoder: Embedding + 2-layer LSTM → latent vector (bottleneck)
- Decoder: Latent vector → 2-layer LSTM → word distribution

Key principle: Train ONLY on genuine reviews (unsupervised).
Anomaly detection via reconstruction error.

Threshold calibration: Use labels only to find optimal threshold,
not for training. The model itself learns no label information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle


class ReviewEncoder(nn.Module):
    """
    LSTM encoder: compresses review into latent vector.
    
    Takes embedded review sequence, processes through 2-layer LSTM,
    returns final hidden state as bottleneck representation.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden state dimension (128)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
    
    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to latent vector.
        
        Args:
            embedded: (batch_size, seq_len, embedding_dim)
            
        Returns:
            latent: (batch_size, hidden_dim) - final hidden state
        """
        _, (hidden, _) = self.lstm(embedded)
        latent = hidden[-1]  # Take last layer's final hidden state
        return latent


class ReviewDecoder(nn.Module):
    """
    LSTM decoder: reconstructs review from latent vector.
    
    Takes latent vector as initial hidden state, generates
    probability distribution over vocabulary at each timestep.
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        """
        Args:
            hidden_dim: LSTM hidden state dimension (128)
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.lstm = nn.LSTM(
            input_size=vocab_size,  # One-hot input
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        latent: torch.Tensor,
        seq_len: int,
        one_hot_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent vector to reconstructed sequence.
        
        Args:
            latent: (batch_size, hidden_dim)
            seq_len: Length of sequences to reconstruct
            one_hot_input: (batch_size, seq_len, vocab_size) one-hot encoded
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Initialize decoder LSTM with encoder latent as hidden state
        h0 = latent.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, hidden)
        c0 = torch.zeros_like(h0)
        
        # Decode
        output, _ = self.lstm(one_hot_input, (h0, c0))  # (batch, seq_len, hidden)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        
        return logits


class ReviewAutoencoder(nn.Module):
    """
    LSTM Autoencoder for unsupervised anomaly detection.
    
    Trained ONLY on genuine reviews. Flags fake reviews by
    measuring reconstruction error on individual tokens.
    
    Key innovation: Combines reconstruction anomaly with
    contradiction score (rating vs sentiment) for robust
    multi-signal fake detection.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension (128)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Shared embedding layer with sentiment model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.encoder = ReviewEncoder(embedding_dim, hidden_dim)
        self.decoder = ReviewDecoder(hidden_dim, vocab_size)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode review, decode to reconstruct.
        
        Args:
            token_ids: (batch_size, seq_len) token indices
            
        Returns:
            reconstructed_logits: (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        embedded = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)
        
        # Encode to latent
        latent = self.encoder(embedded)  # (batch, hidden_dim)
        
        # Create one-hot input for decoder
        seq_len = token_ids.shape[1]
        one_hot = F.one_hot(
            token_ids * (token_ids > 0),  # Mask padding
            num_classes=self.vocab_size
        ).float()  # (batch, seq_len, vocab_size)
        
        # Decode
        reconstructed_logits = self.decoder(latent, seq_len, one_hot)
        
        return reconstructed_logits
    
    def compute_reconstruction_error(
        self,
        token_ids: torch.Tensor,
        reconstructed_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction error (CrossEntropy loss).
        
        Args:
            token_ids: (batch_size, seq_len) original tokens
            reconstructed_logits: (batch_size, seq_len, vocab_size)
            mask: (batch_size, seq_len) to mask padding tokens
            
        Returns:
            error: (batch_size,) mean error per sequence
        """
        batch_size, seq_len = token_ids.shape
        
        # Reshape for CrossEntropyLoss
        logits_flat = reconstructed_logits.reshape(-1, self.vocab_size)
        tokens_flat = token_ids.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, tokens_flat, reduction='none')
        loss = loss.reshape(batch_size, seq_len)
        
        # Mask padding tokens
        if mask is not None:
            loss = loss * mask
            error = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            error = loss.mean(dim=1)
        
        return error
    
    def is_anomaly(self, error: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Classify review as anomalous if error exceeds threshold.
        
        Args:
            error: (batch_size,) reconstruction errors
            threshold: Threshold from calibration set
            
        Returns:
            is_fake: (batch_size,) boolean tensor
        """
        return error > threshold
    
    def save_model(self, filepath: str, threshold: float = None, config: Dict = None) -> None:
        """
        Save model, threshold, and configuration.
        
        Args:
            filepath: Path to save model
            threshold: Learned anomaly threshold
            config: Configuration dict
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'threshold': threshold,
            'config': config
        }
        
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_model(filepath: str, device: str = 'cpu') -> 'ReviewAutoencoder':
        """
        Load autoencoder checkpoint.
        
        Args:
            filepath: Path to saved checkpoint
            device: Device to load on
            
        Returns:
            ReviewAutoencoder with loaded weights
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        model = ReviewAutoencoder(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model
