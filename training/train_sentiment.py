"""
Stage 1 Training: LSTM Sentiment Classifier
=============================================
Trains a 4-layer LSTM + attention model to classify review sentiment
(positive / negative).  Expected validation accuracy: 89–92 %.

Training data: Amazon Reviews dataset (100K positive + 100K negative)
If the dataset is unavailable, a synthetic fallback is used (demo only).

Usage
-----
    cd <project_root>
    ./venv/bin/python training/train_sentiment.py
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.sentiment_model import SentimentLSTM
from utils.preprocessing import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {DEVICE}")

TRAIN_CFG: Dict[str, Any] = {
    'vocab_size'    : 20_000,
    'embedding_dim' : 128,
    'hidden_dim'    : 256,
    'num_layers'    : 4,
    'dropout'       : 0.3,
    'fc_hidden'     : 64,
    'batch_size'    : 64,
    'epochs'        : 15,
    'lr'            : 1e-3,
    'weight_decay'  : 1e-4,
    'patience'      : 4,
    'grad_clip'     : 1.0,
    'max_seq_len'   : 200,
    'val_split'     : 0.10,
    'n_samples'     : 200_000,   # max samples to use from Amazon dataset
    'data_path'     : os.path.join(ROOT_DIR, 'data', 'amazon_reviews_train.csv'),
    'save_dir'      : os.path.join(ROOT_DIR, 'models', 'saved'),
}


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_amazon_data(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load Amazon reviews.  Returns (texts, labels) where label=1→positive."""
    import pandas as pd

    path = cfg['data_path']
    if not os.path.exists(path):
        log.warning(f"Amazon dataset not found at {path} — using synthetic data.")
        return _synthetic_data(cfg['n_samples'])

    log.info(f"Loading Amazon reviews from {path} …")
    df = pd.read_csv(path, header=None, names=['label', 'title', 'text'],
                     nrows=cfg['n_samples'])

    # Amazon format: label 1 = negative, 2 = positive
    texts  = (df['title'].fillna('') + ' ' + df['text'].fillna('')).astype(str).values
    labels = (df['label'] == 2).astype(int).values   # 1 = positive

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    log.info(f"  Loaded {len(labels)} samples: {n_pos} positive, {n_neg} negative")
    return texts, labels


def _synthetic_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    log.warning("⚠  Using SYNTHETIC training data — for demonstration only!")
    pos = [
        "This product is amazing and works perfectly.",
        "I love this item, great quality and fast shipping.",
        "Exceeded my expectations, highly recommend to everyone.",
        "Five stars, will definitely buy again.",
        "Perfect product, exactly as described.",
    ]
    neg = [
        "Terrible quality, broke after one day.",
        "Complete waste of money, do not buy.",
        "Very disappointed, nothing like the pictures.",
        "Worst purchase I have ever made.",
        "Total garbage, stopped working immediately.",
    ]
    rng    = np.random.default_rng(42)
    texts  = [rng.choice(pos if i < n // 2 else neg) for i in range(n)]
    labels = np.array([1] * (n // 2) + [0] * (n - n // 2), dtype=int)
    idx    = rng.permutation(n)
    return np.array(texts)[idx], labels[idx]


# ════════════════════════════════════════════════════════════════════════════
# Training helpers
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, cfg) -> float:
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.float().to(DEVICE)
        optimizer.zero_grad()
        prob, _ = model(X_b)
        loss    = criterion(prob.squeeze(-1), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total, all_probs, all_labels = 0.0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.float().to(DEVICE)
        prob, _  = model(X_b)
        loss     = criterion(prob.squeeze(-1), y_b)
        total   += loss.item()
        all_probs.extend(prob.squeeze(-1).cpu().tolist())
        all_labels.extend(y_b.cpu().tolist())
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)
    return total / max(len(loader), 1), acc


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg      = TRAIN_CFG
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    texts, labels = load_amazon_data(cfg)

    log.info("Building vocabulary …")
    preprocessor = TextPreprocessor(max_vocab=cfg['vocab_size'])
    preprocessor.build_vocabulary(
        [preprocessor.clean_text(t) for t in texts]
    )
    vocab_path = os.path.join(save_dir, 'vocabulary.pkl')
    preprocessor.save_vocabulary(vocab_path)
    log.info(f"  Vocabulary ({preprocessor.vocab_size} tokens) → {vocab_path}")

    log.info("Encoding texts …")
    X = np.array([
        preprocessor.encode_text(preprocessor.clean_text(t), max_len=cfg['max_seq_len'])
        for t in texts
    ])
    X_tensor = torch.LongTensor(X)
    y_tensor = torch.LongTensor(labels)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tensor, y_tensor, test_size=cfg['val_split'],
        stratify=y_tensor, random_state=42,
    )

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),  batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg['batch_size'], shuffle=False)

    log.info(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    model = SentimentLSTM(
        vocab_size    = cfg['vocab_size'],
        embedding_dim = cfg['embedding_dim'],
        hidden_dim    = cfg['hidden_dim'],
        num_layers    = cfg['num_layers'],
        dropout       = cfg['dropout'],
        fc_hidden     = cfg['fc_hidden'],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_acc, patience_ctr = 0.0, 0
    best_path = os.path.join(save_dir, 'sentiment_model_adam_best.pt')
    training_log = []

    log.info(f"\n{'='*60}")
    log.info("  Stage 1 Training — LSTM Sentiment Classifier")
    log.info(f"{'='*60}")

    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss          = train_epoch(model, train_loader, optimizer, criterion, cfg)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        log.info(
            f"  Epoch {epoch:02d}/{cfg['epochs']}  "
            f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        training_log.append(dict(epoch=epoch, train=tr_loss, val=val_loss, acc=val_acc))

        if val_acc > best_acc:
            best_acc     = val_acc
            patience_ctr = 0
            model.save_model(best_path)
            log.info(f"  ✓ Best model saved (val_acc={best_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    log_path = os.path.join(save_dir, 'training_log_sentiment.json')
    with open(log_path, 'w') as f:
        json.dump({'adam': training_log, 'best_val_accuracy': best_acc}, f, indent=2)

    print(f"\n✅ Stage 1 Training Complete!")
    print(f"   Best val accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()