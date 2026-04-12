"""
Text Preprocessing Module for ReviewGuard

Handles text cleaning, tokenization, vocabulary building, encoding/decoding,
sentiment label mapping, and contradiction score computation.

Corresponds to: Practical 1-2 (Data Preprocessing)
"""

import re
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TextPreprocessor:
    """
    Complete text preprocessing pipeline for review analysis.
    
    Provides text cleaning, vocabulary management, encoding/decoding sequences,
    sentiment label mapping, and novel contradiction score computation.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.word2idx = None
        self.idx2word = None
        self.vocab_size = 0
    
    def clean_text(self, text: str) -> str:
        """
        Clean review text for processing.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text (lowercase, no HTML/URLs, special chars removed,
            but punctuation ! and ? preserved as sentiment signals)
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Keep ! and ? as sentiment signals, remove other special chars
        text = re.sub(r'[^a-z0-9\s!?\-\']', '', text)
        
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def build_vocabulary(self, texts: List[str], max_vocab: int = 20000) -> Tuple[Dict, Dict]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of cleaned review texts
            max_vocab: Maximum vocabulary size (top words kept)
            
        Returns:
            Tuple of (word2idx dict, idx2word dict)
            
        Special tokens:
            - 0: <PAD> (padding token)
            - 1: <UNK> (unknown word)
            - 2: <START> (sequence start)
            - 3: <END> (sequence end)
        """
        # Tokenize all texts and count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Get top max_vocab words
        top_words = [word for word, _ in word_counts.most_common(max_vocab - 4)]
        
        # Build dictionaries with special tokens
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        for idx, word in enumerate(top_words, start=4):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        return self.word2idx, self.idx2word
    
    def encode_text(self, text: str, max_len: int = 200) -> np.ndarray:
        """
        Encode text to sequence of indices.
        
        Args:
            text: Cleaned review text
            max_len: Maximum sequence length (pad or truncate)
            
        Returns:
            Numpy array of shape (max_len,) with token indices
            
        OOV (out-of-vocabulary) words mapped to <UNK> (index 1).
        Sequence is padded with <PAD> (index 0) to reach max_len.
        """
        if self.word2idx is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        words = text.split()
        
        # Convert words to indices
        indices = []
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                # Use <UNK> for out-of-vocabulary words
                indices.append(self.word2idx['<UNK>'])
        
        # Truncate if longer than max_len
        if len(indices) > max_len:
            indices = indices[:max_len]
        
        # Pad with <PAD> tokens to reach max_len
        else:
            indices = indices + [self.word2idx['<PAD>']] * (max_len - len(indices))
        
        return np.array(indices, dtype=np.int64)
    
    def decode_text(self, indices: np.ndarray) -> str:
        """
        Decode sequence of indices back to text.
        
        Args:
            indices: Sequence of token indices
            
        Returns:
            Decoded text with special tokens removed
        """
        if self.idx2word is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        words = []
        for idx in indices:
            idx = int(idx)
            if idx in self.idx2word:
                word = self.idx2word[idx]
                # Skip padding and special tokens
                if word not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def map_sentiment_label(self, star_rating: int) -> Optional[int]:
        """
        Map star rating to binary sentiment label.
        
        Args:
            star_rating: Rating from 1-5
            
        Returns:
            0 (Negative) for 1-2 stars
            1 (Positive) for 4-5 stars
            None (skip) for 3 stars (ambiguous)
        """
        if star_rating in [1, 2]:
            return 0  # Negative
        elif star_rating in [4, 5]:
            return 1  # Positive
        else:
            return None  # Skip ambiguous 3-star reviews
    
    def compute_contradiction_score(
        self,
        star_rating: int,
        sentiment_confidence: float,
        sentiment_label: int
    ) -> float:
        """
        Compute contradiction score: novel feature combining rating vs sentiment.
        
        NOVEL CONTRIBUTION (key innovation):
        ────────────────────────────────────
        This is the core signal for detecting rating manipulation attacks:
        - High star rating + negative sentiment words = likely fake
        - Low star rating + positive sentiment words = likely fake
        
        Standard fake review detectors use only reconstruction error.
        ReviewGuard fuses this with contradiction detection, creating
        a two-signal approach not seen in prior papers.
        
        Args:
            star_rating: Product rating (1-5)
            sentiment_confidence: Confidence from sentiment model (0-1)
            sentiment_label: Binary sentiment (0=negative, 1=positive)
            
        Returns:
            Contradiction score 0-1. Higher = more suspicious.
            
        Logic:
            - If 4-5 stars with negative sentiment: very suspicious
              score = (star_rating/5) * sentiment_confidence
            - If 1-2 stars with positive sentiment: very suspicious
              score = (1 - star_rating/5) * sentiment_confidence
            - Otherwise: no contradiction = 0
        """
        # Check for high rating + negative sentiment
        if star_rating >= 4 and sentiment_label == 0:
            # Score increases with: (1) higher rating, (2) higher confidence
            score = (star_rating / 5.0) * sentiment_confidence
            return min(score, 1.0)
        
        # Check for low rating + positive sentiment
        elif star_rating <= 2 and sentiment_label == 1:
            # Score increases with: (1) lower rating, (2) higher confidence
            score = (1.0 - star_rating / 5.0) * sentiment_confidence
            return min(score, 1.0)
        
        # No contradiction
        else:
            return 0.0
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to disk for reproducibility.
        
        Args:
            filepath: Path to save vocabulary pickle file
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from disk.
        
        Args:
            filepath: Path to saved vocabulary pickle file
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = vocab_data['vocab_size']
