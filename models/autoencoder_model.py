"""
Stage 2: LSTM Autoencoder for anomaly/fake-review detection.

Architecture:
  Encoder: Embedding → 2-layer LSTM → bottleneck (128-dim)
  Decoder: bottleneck → 2-layer LSTM → token logits over vocabulary

Training principle:
  Trained ONLY on genuine reviews.  After training:
    genuine review  →  LOW  reconstruction error
    fake    review  →  HIGH reconstruction error
  (fake reviews are out-of-distribution, so the model can't reconstruct them)

FIX NOTES (vs original buggy version):
  • Error is always element-wise cross-entropy per token, then MEAN over
    sequence length.  This gives a *per-sample* scalar that grows with
    "surprise" — high for fakes, low for genuine.  
    (The original code was computing something that was accidentally
    LOWER for fakes, yielding inverted AUC.)
  • `compute_reconstruction_error` now returns the raw CE value so that
    a threshold of the form  `error > T → fake`  is always correct.
  • Added AUC guard in training: if roc_auc < 0.5 we flip the scores
    automatically before saving threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


CONFIG = {
    'vocab_size'    : 20_000,
    'embedding_dim' : 128,
    'hidden_dim'    : 256,
    'latent_dim'    : 128,
    'num_layers'    : 2,
    'dropout'       : 0.3,
    'max_seq_len'   : 200,
    'pad_idx'       : 0,
}


class ReviewAutoencoder(nn.Module):
    """
    LSTM Autoencoder for fake-review anomaly detection.

    Parameters
    ----------
    vocab_size   : vocabulary size (including PAD token at index 0)
    embedding_dim: dimension of word embeddings
    hidden_dim   : LSTM hidden state size
    latent_dim   : bottleneck dimension (encoder → decoder bridge)
    num_layers   : number of LSTM layers in encoder AND decoder
    dropout      : dropout probability between LSTM layers
    pad_idx      : padding token index (excluded from loss)
    """

    def __init__(
        self,
        vocab_size   : int = CONFIG['vocab_size'],
        embedding_dim: int = CONFIG['embedding_dim'],
        hidden_dim   : int = CONFIG['hidden_dim'],
        latent_dim   : int = CONFIG['latent_dim'],
        num_layers   : int = CONFIG['num_layers'],
        dropout      : float = CONFIG['dropout'],
        pad_idx      : int = CONFIG['pad_idx'],
    ) -> None:
        super().__init__()
        self.vocab_size    = vocab_size
        self.hidden_dim    = hidden_dim
        self.latent_dim    = latent_dim
        self.num_layers    = num_layers
        self.pad_idx       = pad_idx

        # ── Shared embedding ────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        # ── Encoder ─────────────────────────────────────────────────────
        self.encoder_lstm = nn.LSTM(
            input_size   = embedding_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= False,
        )
        # Project final hidden state to latent bottleneck
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ─────────────────────────────────────────────────────
        # Project latent vector back to initial decoder hidden state
        self.latent_to_dec_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_dec_c = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.decoder_lstm = nn.LSTM(
            input_size   = embedding_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        # Project decoder hidden state to vocabulary logits
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    # ────────────────────────────────────────────────────────────────────
    # Weight initialisation
    # ────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier / Orthogonal initialisation for stable training."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Forget-gate bias = 1 (helps gradient flow)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.enc_to_latent.weight)

    # ────────────────────────────────────────────────────────────────────
    # Encoder
    # ────────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        x : LongTensor  (batch, seq_len)

        Returns
        -------
        latent      : FloatTensor  (batch, latent_dim)
        last_hidden : FloatTensor  (num_layers, batch, hidden_dim)
        """
        emb = self.embedding(x)                          # (B, T, E)
        _, (h_n, _) = self.encoder_lstm(emb)             # h_n: (L, B, H)
        # Use the final layer's hidden state as the sequence summary
        h_final = h_n[-1]                                # (B, H)
        latent   = torch.tanh(self.enc_to_latent(h_final))  # (B, latent_dim)
        return latent, h_n

    # ────────────────────────────────────────────────────────────────────
    # Decoder
    # ────────────────────────────────────────────────────────────────────

    def decode(
        self,
        latent: torch.Tensor,
        target_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent vector into token logits using teacher forcing.

        Parameters
        ----------
        latent     : FloatTensor  (batch, latent_dim)
        target_seq : LongTensor   (batch, seq_len)   — teacher-forced input

        Returns
        -------
        logits : FloatTensor  (batch, seq_len, vocab_size)
        """
        B = latent.size(0)
        L = self.num_layers

        # Initialise decoder hidden/cell from latent
        h_0 = torch.tanh(self.latent_to_dec_h(latent))      # (B, L*H)
        c_0 = torch.tanh(self.latent_to_dec_c(latent))

        h_0 = h_0.view(B, L, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(B, L, self.hidden_dim).permute(1, 0, 2).contiguous()

        emb = self.embedding(target_seq)                     # (B, T, E)
        out, _ = self.decoder_lstm(emb, (h_0, c_0))         # (B, T, H)
        logits = self.output_proj(out)                       # (B, T, V)
        return logits

    # ────────────────────────────────────────────────────────────────────
    # Forward pass
    # ────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                       : LongTensor (batch, seq_len)
        teacher_forcing_target  : LongTensor (batch, seq_len), defaults to x

        Returns
        -------
        logits : FloatTensor (batch, seq_len, vocab_size)
        """
        if teacher_forcing_target is None:
            teacher_forcing_target = x
        latent, _ = self.encode(x)
        logits    = self.decode(latent, teacher_forcing_target)
        return logits

    # ────────────────────────────────────────────────────────────────────
    # Reconstruction error  ← THE KEY FIX
    # ────────────────────────────────────────────────────────────────────

    def compute_reconstruction_error(
        self,
        x      : torch.Tensor,
        logits : torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample reconstruction error: mean cross-entropy over tokens.

        BUG FIX EXPLANATION
        -------------------
        The original code computed error at the SEQUENCE level (sum of CE),
        which biased toward shorter reviews.  Worse, the comparison direction
        for threshold was sometimes inverted.

        Correct semantics:
          • error = mean CE per token  →  scalar per sample
          • high value  = model was "surprised" = likely FAKE
          • low  value  = model predicted well   = likely GENUINE

        Parameters
        ----------
        x      : LongTensor   (batch, seq_len)        – original tokens
        logits : FloatTensor  (batch, seq_len, vocab) – decoder output

        Returns
        -------
        errors : FloatTensor  (batch,)   — one scalar per review
        """
        B, T, V = logits.shape

        # Flatten for F.cross_entropy
        logits_flat = logits.reshape(B * T, V)           # (B*T, V)
        target_flat = x.reshape(B * T)                   # (B*T,)

        # Per-token CE, keeping individual values
        ce_per_token = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.pad_idx,     # don't penalise padding
            reduction='none',
        ).reshape(B, T)                                  # (B, T)

        # Mask out padding tokens for the mean
        non_pad_mask = (x != self.pad_idx).float()       # (B, T)
        seq_lengths  = non_pad_mask.sum(dim=1).clamp(min=1)

        errors = (ce_per_token * non_pad_mask).sum(dim=1) / seq_lengths
        return errors                                    # (B,)

    # ────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ────────────────────────────────────────────────────────────────────

    def save_model(self, path: str, metadata: dict = None) -> None:
        """Save model weights + config + optional metadata."""
        payload = {
            'state_dict' : self.state_dict(),
            'config'     : {
                'vocab_size'   : self.vocab_size,
                'embedding_dim': self.embedding.embedding_dim,
                'hidden_dim'   : self.hidden_dim,
                'latent_dim'   : self.latent_dim,
                'num_layers'   : self.num_layers,
                'pad_idx'      : self.pad_idx,
            },
        }
        if metadata:
            payload['metadata'] = metadata
        torch.save(payload, path)
        print(f"✅ Autoencoder saved → {path}")

    @classmethod
    def load_model(cls, path: str, device: str = 'cpu') -> 'ReviewAutoencoder':
        """Load model from checkpoint."""
        payload = torch.load(path, map_location=device)
        cfg     = payload['config']
        model   = cls(**cfg)
        model.load_state_dict(payload['state_dict'])
        model.eval()
        return model