"""
Evaluation Utilities
====================
Computes and displays all metrics for Stage 1 (sentiment) and
Stage 2 (autoencoder) models.

FIX NOTES
---------
• Threshold is loaded from `threshold_config.json` which now stores the
  `scores_inverted` flag.  Predictions use the correct direction:
      predicted_fake = (error > threshold)   [after optional negation]
• AUC is always computed against the raw (or corrected) error, not the
  binary prediction, for a proper ROC curve.
• Added a standalone CLI so you can run:
      python training/evaluate.py --stage autoencoder
"""

import os
import sys
import json
import argparse
import pickle
import logging
from typing import Optional, Dict

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: Dict[str, float], title: str = 'Metrics') -> None:
    bar = '=' * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")
    print(bar + "\n")


def load_threshold_config(save_dir: str) -> Dict:
    """Load threshold + inversion flag saved during calibration."""
    path = os.path.join(save_dir, 'threshold_config.json')
    if not os.path.exists(path):
        log.warning(f"threshold_config.json not found at {path}. Using default T=0.5.")
        return {'threshold': 0.5, 'scores_inverted': False}
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Sentiment Evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate_sentiment(
    save_dir  : str,
    data_path : str = None,
    n_samples : int = 10_000,
) -> Dict[str, float]:
    """
    Evaluate the Stage 1 sentiment classifier.

    Returns a dict of accuracy / precision / recall / F1.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score,
    )
    from models.sentiment_model import SentimentLSTM
    from utils.preprocessing import TextPreprocessor

    model_path = os.path.join(save_dir, 'sentiment_model_adam_best.pt')
    vocab_path = os.path.join(save_dir, 'vocabulary.pkl')

    if not os.path.exists(model_path):
        log.error(f"Sentiment model not found: {model_path}")
        return {}
    if not os.path.exists(vocab_path):
        log.error(f"Vocabulary not found: {vocab_path}")
        return {}

    model        = SentimentLSTM.load_model(model_path, device=str(DEVICE)).to(DEVICE)
    preprocessor = TextPreprocessor()
    preprocessor.load_vocabulary(vocab_path)

    # Try to load Amazon reviews test set
    amazon_path = data_path or os.path.join(ROOT_DIR, 'data', 'amazon_reviews_test.csv')
    if not os.path.exists(amazon_path):
        log.warning("Test data not found — generating synthetic evaluation set.")
        texts, labels = _synthetic_sentiment_data(n_samples)
    else:
        import pandas as pd
        df     = pd.read_csv(amazon_path).sample(min(n_samples, len(pd.read_csv(amazon_path))), random_state=42)
        texts  = df['text'].fillna('').astype(str).values
        labels = (df['stars'] >= 4).astype(int).values   # 4-5★ = positive

    probs = _batch_sentiment_predict(model, preprocessor, texts)
    preds = (probs >= 0.5).astype(int)

    metrics = dict(
        accuracy  = accuracy_score(labels, preds),
        precision = precision_score(labels, preds, zero_division=0),
        recall    = recall_score(labels, preds, zero_division=0),
        f1        = f1_score(labels, preds, zero_division=0),
        auc_roc   = roc_auc_score(labels, probs),
    )
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm.tolist()

    print_metrics_table(
        {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        title='Stage 1 — Sentiment Evaluation Results',
    )
    print(f"  Confusion Matrix:\n{cm}\n")
    return metrics


def _batch_sentiment_predict(model, preprocessor, texts, batch=128):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            encoded     = np.array([
                preprocessor.encode_text(preprocessor.clean_text(t), max_len=200)
                for t in batch_texts
            ])
            x    = torch.LongTensor(encoded).to(DEVICE)
            prob, _ = model(x)
            all_probs.append(prob.squeeze(-1).cpu().numpy())
    return np.concatenate(all_probs)


def _synthetic_sentiment_data(n):
    """Very simple synthetic data for smoke-testing."""
    rng = np.random.default_rng(0)
    texts  = ['great product' if i % 2 == 0 else 'terrible product' for i in range(n)]
    labels = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=int)
    return texts, labels


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Autoencoder Evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate_autoencoder(
    save_dir  : str,
    data_path : str = None,
) -> Dict[str, float]:
    """
    Evaluate the Stage 2 autoencoder fake-detector on the mexwell corpus.

    Uses the threshold and inversion flag saved in threshold_config.json.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score,
    )
    from models.autoencoder_model import ReviewAutoencoder
    from utils.preprocessing import TextPreprocessor
    from training.train_autoencoder import (
        load_mexwell_data, encode_texts, compute_reconstruction_errors,
        TRAIN_CFG,
    )

    model_path = os.path.join(save_dir, 'autoencoder_checkpoint.pt')
    vocab_path = os.path.join(save_dir, 'vocabulary_ae.pkl')

    if not os.path.exists(model_path):
        log.error(f"Autoencoder model not found: {model_path}")
        return {}

    cfg         = dict(TRAIN_CFG)
    if data_path:
        cfg['data_path'] = data_path

    model        = ReviewAutoencoder.load_model(model_path, device=str(DEVICE)).to(DEVICE)
    preprocessor = TextPreprocessor(max_vocab=cfg['vocab_size'])

    if os.path.exists(vocab_path):
        preprocessor.load_vocabulary(vocab_path)
    else:
        log.warning("vocabulary_ae.pkl not found — building from scratch.")
        texts, labels = load_mexwell_data(cfg)
        preprocessor.build_vocabulary(texts[labels == 0].tolist())

    texts, labels = load_mexwell_data(cfg)
    X, y          = encode_texts(texts, labels, cfg, preprocessor)
    errors        = compute_reconstruction_errors(model, X)

    thr_cfg   = load_threshold_config(save_dir)
    threshold = thr_cfg['threshold']
    inverted  = thr_cfg.get('scores_inverted', False)

    if inverted:
        log.warning("Applying score inversion (loaded from threshold_config.json)")
        errors = -errors

    labels_np = y.numpy()
    preds     = (errors > threshold).astype(int)

    metrics = dict(
        accuracy  = accuracy_score(labels_np, preds),
        precision = precision_score(labels_np, preds, zero_division=0),
        recall    = recall_score(labels_np, preds, zero_division=0),
        f1        = f1_score(labels_np, preds, zero_division=0),
        auc_roc   = roc_auc_score(labels_np, errors),
        threshold = threshold,
    )
    cm = confusion_matrix(labels_np, preds)
    metrics['confusion_matrix'] = cm.tolist()

    print_metrics_table(
        {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        title='Stage 2 — Autoencoder Evaluation Results',
    )
    print(f"  Confusion Matrix:\n{cm}\n")
    return metrics


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate ReviewGuard models')
    parser.add_argument('--stage', choices=['sentiment', 'autoencoder', 'both'],
                        default='both')
    parser.add_argument('--save_dir', default=os.path.join(ROOT_DIR, 'models', 'saved'))
    args = parser.parse_args()

    if args.stage in ('sentiment', 'both'):
        evaluate_sentiment(args.save_dir)
    if args.stage in ('autoencoder', 'both'):
        evaluate_autoencoder(args.save_dir)


if __name__ == '__main__':
    main()