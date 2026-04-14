"""
Text Preprocessing Utilities
==============================
Provides tokenisation, vocabulary building, and integer encoding for
all models in ReviewGuard.

FIX NOTES (vs original)
------------------------
1.  `build_vocabulary` now sorts by frequency descending before
    truncating to `max_vocab`.  The original code used dict iteration
    order which is random → different runs produce different vocabularies,
    making saved models non-reproducible.

2.  PAD=0, UNK=1 are always the first two entries (consistent with
    `padding_idx=0` in all models).

3.  `encode_text` respects `max_len` with proper truncation AND left-
    padding so the most recent tokens are seen by the LSTM.

4.  `compute_contradiction_score` moved to `utils/fusion.py`
    (ContradictionScorer).  The method here remains as a thin wrapper for
    backward compatibility.
"""

import re
import pickle
import string
from collections import Counter
from typing import List, Optional

import numpy as np

# Safe NLTK import
try:
    import nltk
    from nltk.tokenize import word_tokenize
    _USE_NLTK = True
except ImportError:
    _USE_NLTK = False

try:
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words('english'))
except Exception:
    _STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'he', 'she', 'it', 'they', 'them', 'what',
        'which', 'who', 'whom', 'this', 'that', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet',
        'so', 'at', 'by', 'in', 'of', 'on', 'to', 'up', 'as',
        'is', 'it', 'its', 'with', 'from', 'into', 'through',
    }

# Special tokens
PAD_TOKEN = '<PAD>'   # index 0
UNK_TOKEN = '<UNK>'   # index 1
PAD_IDX   = 0
UNK_IDX   = 1


class TextPreprocessor:
    """
    Full text preprocessing pipeline:
      clean_text → tokenize → build_vocabulary → encode_text

    Parameters
    ----------
    max_vocab : int
        Maximum vocabulary size (including PAD and UNK).
    remove_stopwords : bool
        Whether to strip stop-words during cleaning.
        Default False — removing stop-words hurts LSTM models because
        they help determine sentiment context ("not good", "never worked").
    """

    def __init__(
        self,
        max_vocab       : int  = 20_000,
        remove_stopwords: bool = False,
    ) -> None:
        self.max_vocab        = max_vocab
        self.remove_stopwords = remove_stopwords
        self.word2idx: dict   = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: dict   = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.vocab_built      = False

    # ──────────────────────────────────────────────────────────────────
    # Text cleaning
    # ──────────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Normalise a raw review string.

        Steps
        -----
        1. Lowercase
        2. Expand contractions (don't → do not, etc.)
        3. Remove URLs
        4. Remove HTML tags
        5. Remove non-alphanumeric characters (keep spaces)
        6. Collapse multiple spaces
        7. Optionally remove stop-words
        """
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = self._expand_contractions(text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)      # URLs
        text = re.sub(r'<[^>]+>', ' ', text)                     # HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)                 # punctuation
        text = re.sub(r'\s+', ' ', text).strip()

        if self.remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in _STOP_WORDS]
            text   = ' '.join(tokens)

        return text

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "i'm": "i am", "i've": "i have",
            "i'll": "i will", "i'd": "i would", "you're": "you are",
            "you've": "you have", "you'll": "you will", "you'd": "you would",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are", "that's": "that is",
            "there's": "there is", "here's": "here is",
        }
        for c, e in contractions.items():
            text = text.replace(c, e)
        return text

    # ──────────────────────────────────────────────────────────────────
    # Tokenisation
    # ──────────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """Split cleaned text into word tokens."""
        if _USE_NLTK:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return text.split()

    # ──────────────────────────────────────────────────────────────────
    # Vocabulary building  (FIX: deterministic + frequency-sorted)
    # ──────────────────────────────────────────────────────────────────

    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build word2idx / idx2word from a list of already-cleaned texts.

        The vocabulary is sorted by descending frequency so the most
        common words always get the lowest indices — this is deterministic
        across runs (the original used dict iteration order which is not).

        PAD (0) and UNK (1) are reserved before any data-driven tokens.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Sort by frequency (desc), then alphabetically for tie-breaking
        sorted_words = sorted(counter.keys(), key=lambda w: (-counter[w], w))

        # Reserve slots for PAD and UNK; fill the rest
        max_data_tokens = self.max_vocab - 2
        vocab_words     = sorted_words[:max_data_tokens]

        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word]  = idx
            self.idx2word[idx]   = word

        self.vocab_built = True

    # ──────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────

    def encode_text(self, text: str, max_len: int = 200) -> np.ndarray:
        """
        Convert a cleaned text string to a fixed-length integer array.

        Tokens beyond `max_len` are truncated from the LEFT (we keep the
        most recent `max_len` tokens — the end of a review is often more
        revealing than the start).

        Shorter sequences are left-padded with PAD (0).

        Returns
        -------
        np.ndarray of shape (max_len,), dtype int64
        """
        tokens  = self.tokenize(text)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]

        # Keep the LAST max_len tokens (most informative for reviews)
        if len(indices) > max_len:
            indices = indices[-max_len:]

        # Left-pad
        pad_len = max_len - len(indices)
        padded  = [PAD_IDX] * pad_len + indices
        return np.array(padded, dtype=np.int64)

    def decode_indices(self, indices: List[int]) -> str:
        """Convert integer indices back to words (for debugging)."""
        words = [self.idx2word.get(i, UNK_TOKEN) for i in indices if i != PAD_IDX]
        return ' '.join(words)

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save_vocabulary(self, path: str) -> None:
        payload = {
            'word2idx'        : self.word2idx,
            'idx2word'        : self.idx2word,
            'max_vocab'       : self.max_vocab,
            'remove_stopwords': self.remove_stopwords,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    def load_vocabulary(self, path: str) -> None:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self.word2idx        = payload['word2idx']
        self.idx2word        = payload['idx2word']
        self.max_vocab       = payload.get('max_vocab', 20_000)
        self.remove_stopwords= payload.get('remove_stopwords', False)
        self.vocab_built     = True

    # ──────────────────────────────────────────────────────────────────
    # Backward-compat wrapper
    # ──────────────────────────────────────────────────────────────────

    def compute_contradiction_score(
        self,
        star_rating     : int,
        sentiment_prob  : float,
        sentiment_label : int,
    ) -> float:
        """
        Thin wrapper kept for backward compatibility.
        New code should use `utils.fusion.ContradictionScorer` directly.
        """
        from utils.fusion import ContradictionScorer
        return ContradictionScorer().score(star_rating, sentiment_prob, sentiment_label)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)