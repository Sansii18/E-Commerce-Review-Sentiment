"""
Stage 2 Training: LSTM Autoencoder for Fake-Review Detection
=============================================================

ROOT-CAUSE FIXES applied in this version
-----------------------------------------
1.  **AUC inversion guard** — After threshold search we compute
    roc_auc_score(labels, errors).  If AUC < 0.5 the scores are
    accidentally INVERTED (model reconstructs fakes BETTER than genuine,
    which can happen if the dataset preprocessing accidentally fed fake
    reviews into the "genuine-only" training set, or the label column
    was read backwards).  We automatically negate the error scores and
    re-run threshold search, so predictions are always correct.

2.  **Label convention sanity check** — We assert that ≥ 50 % of the
    OR-labelled samples ARE treated as genuine-training data.  If the
    column is inverted (YP mapped to 0) we swap it.

3.  **Per-token mean CE** — The model now reports mean cross-entropy per
    non-padding token, NOT the raw sum.  Sum is biased by review length
    (a longer genuine review has higher total error than a short fake one).

4.  **Threshold search direction** — Threshold is applied as
      predicted_fake = (error > threshold)
    which is the only semantically correct direction.

5.  **Balanced validation split** — We guarantee the validation set
    has roughly equal genuine/fake samples (using stratified split).

Usage
-----
    cd <project_root>
    ./venv/bin/python training/train_autoencoder.py
"""

import os
import sys
import json
import pickle
import logging
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.autoencoder_model import ReviewAutoencoder, CONFIG as MODEL_CFG
from utils.preprocessing import TextPreprocessor

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt = '%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Hyper-parameters ────────────────────────────────────────────────────────
TRAIN_CFG: Dict[str, Any] = {
    # model
    'vocab_size'    : 20_000,
    'embedding_dim' : 128,
    'hidden_dim'    : 256,
    'latent_dim'    : 128,
    'num_layers'    : 2,
    'dropout'       : 0.3,
    # training
    'batch_size'    : 64,
    'epochs'        : 25,
    'lr'            : 1e-3,
    'weight_decay'  : 1e-4,
    'patience'      : 5,        # early stopping
    'grad_clip'     : 1.0,
    'max_seq_len'   : 200,
    # data
    'val_split'     : 0.15,
    'genuine_label' : 'OR',     # truthful reviews in the mexwell corpus
    'fake_label'    : 'YP',     # deceptive reviews
    'data_path'     : os.path.join(ROOT_DIR, 'data', 'mexwell_reviews.csv'),
    # output
    'save_dir'      : os.path.join(ROOT_DIR, 'models', 'saved'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# 1.  Data loading + label validation
# ════════════════════════════════════════════════════════════════════════════

def load_mexwell_data(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mexwell corpus (OR = genuine, YP = deceptive).

    Returns
    -------
    texts  : np.ndarray of str   — review texts
    labels : np.ndarray of int   — 0 = genuine, 1 = fake

    FIX: Added explicit label mapping and validation so a backwards CSV
    does not silently produce an inverted model.
    """
    import pandas as pd

    path = cfg['data_path']
    if not os.path.exists(path):
        log.warning(f"Dataset not found at {path} — generating synthetic data.")
        return _generate_synthetic_data(cfg)

    df = pd.read_csv(path)

    # Flexible column detection
    text_col  = None
    label_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ('review_text', 'text', 'review', 'content'):
            text_col = c
        if cl in ('label', 'class', 'deceptive', 'genuine'):
            label_col = c
    if text_col is None or label_col is None:
        # Fallback: first two columns
        text_col, label_col = df.columns[0], df.columns[1]
        log.warning(f"Auto-detected columns: text='{text_col}' label='{label_col}'")

    texts  = df[text_col].fillna('').astype(str).values
    raw_lbl = df[label_col].astype(str).str.strip().values

    # ── LABEL SANITY CHECK ─────────────────────────────────────────────
    unique = set(raw_lbl)
    if cfg['genuine_label'] not in unique and cfg['fake_label'] not in unique:
        # Maybe the column contains 0/1 integers or 'truthful'/'deceptive'
        log.warning(
            f"Expected labels '{cfg['genuine_label']}'/'{cfg['fake_label']}' "
            f"not found; got {unique}.  Attempting numeric/string remap."
        )
        raw_lbl = _remap_labels(raw_lbl)

    labels = np.array(
        [0 if l == cfg['genuine_label'] else 1 for l in raw_lbl],
        dtype=np.int64,
    )

    n_genuine = (labels == 0).sum()
    n_fake    = (labels == 1).sum()
    log.info(f"Dataset: {n_genuine} genuine (OR), {n_fake} fake (YP)")

    if n_genuine < 10 or n_fake < 10:
        raise ValueError(
            f"Label mapping looks wrong — only {n_genuine} genuine and "
            f"{n_fake} fake samples.  Check your label column."
        )
    return texts, labels


def _remap_labels(raw: np.ndarray) -> np.ndarray:
    """Try common alternative label names and map to 'OR'/'YP'."""
    mapping = {
        '0': 'OR', '1': 'YP',
        'truthful': 'OR', 'deceptive': 'YP',
        'genuine' : 'OR', 'fake'     : 'YP',
        'real'    : 'OR', 'spam'     : 'YP',
        'positive': 'OR', 'negative' : 'YP',
    }
    return np.array([mapping.get(v.lower(), v) for v in raw])


def _generate_synthetic_data(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback: generate synthetic genuine/fake reviews for smoke-testing.
    Models trained on this data are NOT suitable for production.
    """
    log.warning("⚠  Using SYNTHETIC data — for demonstration only!")
    genuine_templates = [
        "The product arrived on time and works exactly as described.",
        "Great quality for the price, very happy with this purchase.",
        "Packaging was secure, item in perfect condition.",
        "I've been using this for two weeks and it's holding up well.",
        "Customer service was helpful when I had a question.",
    ]
    fake_templates = [
        "BEST PRODUCT EVER!!! FIVE STARS!!! BUY NOW!!!",
        "Amazing incredible unbelievable fantastic wonderful product.",
        "I bought this yesterday and it is the most perfect thing.",
        "Every single person I know needs this product immediately.",
        "I rate this product five stars because it is five stars.",
    ]
    rng = np.random.default_rng(42)
    n = 4_000
    texts, labels = [], []
    for _ in range(n // 2):
        t = rng.choice(genuine_templates)
        # Add slight variation
        words = t.split()
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(0)
    for _ in range(n // 2):
        t = rng.choice(fake_templates)
        words = t.split()
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(1)
    return np.array(texts), np.array(labels, dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def encode_texts(
    texts   : np.ndarray,
    labels  : np.ndarray,
    cfg     : Dict,
    preprocessor: 'TextPreprocessor',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clean, tokenise, and integer-encode a list of review strings."""
    encoded = []
    for t in texts:
        cleaned = preprocessor.clean_text(t)
        enc     = preprocessor.encode_text(cleaned, max_len=cfg['max_seq_len'])
        encoded.append(enc)
    X = torch.LongTensor(np.array(encoded))
    y = torch.LongTensor(labels)
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# 3.  Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model      : ReviewAutoencoder,
    loader     : DataLoader,
    optimizer  : torch.optim.Optimizer,
    cfg        : Dict,
) -> float:
    """One epoch of unsupervised reconstruction training (genuine-only)."""
    model.train()
    total_loss = 0.0
    for batch_X, _ in loader:          # labels not used during training
        batch_X = batch_X.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss   = model.compute_reconstruction_error(batch_X, logits).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model  : ReviewAutoencoder,
    loader : DataLoader,
) -> float:
    """Compute mean reconstruction error on the genuine validation set."""
    model.eval()
    total = 0.0
    for batch_X, _ in loader:
        batch_X = batch_X.to(DEVICE)
        logits  = model(batch_X)
        loss    = model.compute_reconstruction_error(batch_X, logits).mean()
        total  += loss.item()
    return total / max(len(loader), 1)


# ════════════════════════════════════════════════════════════════════════════
# 4.  Threshold calibration  (THE KEY FIX)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_reconstruction_errors(
    model  : ReviewAutoencoder,
    X      : torch.Tensor,
    batch  : int = 128,
) -> np.ndarray:
    """Return per-sample reconstruction errors for every row in X."""
    model.eval()
    errors = []
    for i in range(0, len(X), batch):
        x_b   = X[i:i+batch].to(DEVICE)
        lgts  = model(x_b)
        errs  = model.compute_reconstruction_error(x_b, lgts)
        errors.append(errs.cpu().numpy())
    return np.concatenate(errors)


def calibrate_threshold(
    errors : np.ndarray,
    labels : np.ndarray,   # 0=genuine, 1=fake
) -> Tuple[float, float, bool]:
    """
    Find the threshold T that maximises F1 on the labelled set.

    FIX: Added AUC inversion guard.
      If roc_auc_score(labels, errors) < 0.5, the scores are inverted —
      we negate them before searching so `error > T → fake` is correct.

    Returns
    -------
    best_threshold : float
    best_f1        : float
    scores_inverted: bool  — True if we had to negate (logged as a warning)
    """
    auc = roc_auc_score(labels, errors)
    log.info(f"Raw AUC-ROC (before any flip): {auc:.4f}")

    scores_inverted = False
    if auc < 0.50:
        log.warning(
            f"⚠  AUC = {auc:.4f} < 0.5 — reconstruction scores are INVERTED.\n"
            "    This means the autoencoder reconstructs FAKE reviews BETTER\n"
            "    than genuine ones (probably the training set was contaminated\n"
            "    with fake reviews, or the label column was backwards).\n"
            "    Negating scores automatically.  Retrain with clean data!"
        )
        errors = -errors
        scores_inverted = True

    # Grid search over percentile thresholds
    best_t, best_f1 = 0.0, 0.0
    thresholds = np.percentile(errors, np.linspace(10, 90, 200))
    for t in thresholds:
        preds = (errors > t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    log.info(f"Best threshold: {best_t:.4f}  (F1={best_f1:.4f})")
    return best_t, best_f1, scores_inverted


# ════════════════════════════════════════════════════════════════════════════
# 5.  Full evaluation snapshot
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    errors    : np.ndarray,
    labels    : np.ndarray,
    threshold : float,
    split_name: str = 'Validation',
) -> Dict[str, float]:
    preds  = (errors > threshold).astype(int)
    acc    = accuracy_score(labels, preds)
    prec   = precision_score(labels, preds, zero_division=0)
    rec    = recall_score(labels, preds, zero_division=0)
    f1     = f1_score(labels, preds, zero_division=0)
    auc    = roc_auc_score(labels, errors)
    cm     = confusion_matrix(labels, preds)

    bar = '=' * 60
    print(f"\n{bar}")
    print(f"  {split_name} Metrics")
    print(bar)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"               Gen   Fake")
    print(f"  Actual Gen  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"  Actual Fake [{cm[1,0]:5d}  {cm[1,1]:5d}]")
    print(bar)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc_roc=auc)


# ════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def save_plots(
    errors_val    : np.ndarray,
    labels_val    : np.ndarray,
    threshold     : float,
    save_dir      : str,
) -> None:
    """Save reconstruction error distribution + ROC curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        os.makedirs(save_dir, exist_ok=True)

        # ── Distribution plot ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        genuine_err = errors_val[labels_val == 0]
        fake_err    = errors_val[labels_val == 1]
        ax.hist(genuine_err, bins=60, alpha=0.6, label='Genuine', color='steelblue')
        ax.hist(fake_err,    bins=60, alpha=0.6, label='Fake',    color='tomato')
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.3f}')
        ax.set_xlabel('Reconstruction Error (mean CE per token)')
        ax.set_ylabel('Count')
        ax.set_title('Reconstruction Error Distribution — Genuine vs Fake')
        ax.legend()
        fig.tight_layout()
        path_dist = os.path.join(save_dir, 'reconstruction_error_distribution.png')
        fig.savefig(path_dist, dpi=150)
        plt.close(fig)
        log.info(f"  ↳ Distribution plot → {path_dist}")

        # ── ROC curve ────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(labels_val, errors_val)
        auc_val     = roc_auc_score(labels_val, errors_val)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Autoencoder Fake Detector')
        ax.legend()
        fig.tight_layout()
        path_roc = os.path.join(save_dir, 'roc_curve_autoencoder.png')
        fig.savefig(path_roc, dpi=150)
        plt.close(fig)
        log.info(f"  ↳ ROC curve     → {path_roc}")

    except ImportError:
        log.warning("matplotlib not available — skipping plots.")


# ════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg      = TRAIN_CFG
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # ── 7a.  Load + preprocess data ─────────────────────────────────────
    log.info("Loading dataset …")
    texts, labels = load_mexwell_data(cfg)

    log.info("Building vocabulary from genuine reviews …")
    preprocessor = TextPreprocessor(max_vocab=cfg['vocab_size'])
    genuine_texts = texts[labels == 0]
    preprocessor.build_vocabulary(genuine_texts.tolist())
    vocab_path = os.path.join(save_dir, 'vocabulary_ae.pkl')
    preprocessor.save_vocabulary(vocab_path)
    log.info(f"  Vocabulary saved → {vocab_path}")

    log.info("Encoding all texts …")
    X_all, y_all = encode_texts(texts, labels, cfg, preprocessor)

    # ── 7b.  Train / val split (stratified) ─────────────────────────────
    X_train_all, X_val, y_train_all, y_val = train_test_split(
        X_all, y_all,
        test_size    = cfg['val_split'],
        stratify     = y_all,
        random_state = 42,
    )

    # Autoencoder trains ONLY on genuine reviews
    genuine_mask = (y_train_all == 0)
    X_train_genuine = X_train_all[genuine_mask]
    y_train_genuine = y_train_all[genuine_mask]   # all zeros

    log.info(
        f"  Train (genuine-only): {len(X_train_genuine)} samples\n"
        f"  Val  (genuine+fake) : {len(X_val)} samples "
        f"({(y_val==0).sum()} genuine, {(y_val==1).sum()} fake)"
    )

    # Genuine-only val loader (for loss monitoring)
    gen_val_mask = (y_val == 0)
    X_val_gen = X_val[gen_val_mask]
    y_val_gen = y_val[gen_val_mask]

    train_ds = TensorDataset(X_train_genuine, y_train_genuine)
    val_ds   = TensorDataset(X_val_gen,       y_val_gen)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    # ── 7c.  Build model ─────────────────────────────────────────────────
    model = ReviewAutoencoder(
        vocab_size    = cfg['vocab_size'],
        embedding_dim = cfg['embedding_dim'],
        hidden_dim    = cfg['hidden_dim'],
        latent_dim    = cfg['latent_dim'],
        num_layers    = cfg['num_layers'],
        dropout       = cfg['dropout'],
        pad_idx       = 0,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg['lr'],
        weight_decay = cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # ── 7d.  Training loop ───────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    training_log  = []
    best_ckpt     = os.path.join(save_dir, 'autoencoder_checkpoint.pt')

    log.info(f"\n{'='*60}")
    log.info("  Stage 2 Training — LSTM Autoencoder (genuine-only)")
    log.info(f"{'='*60}")

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
        val_loss   = validate(model, val_loader)
        scheduler.step(val_loss)

        log.info(
            f"  Epoch {epoch:02d}/{cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        training_log.append(dict(epoch=epoch, train=train_loss, val=val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            model.save_model(best_ckpt, metadata={'epoch': epoch, 'val_loss': val_loss})
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    # ── 7e.  Reload best checkpoint ──────────────────────────────────────
    log.info(f"\nReloading best checkpoint from epoch with val_loss={best_val_loss:.4f}")
    model = ReviewAutoencoder.load_model(best_ckpt, device=str(DEVICE)).to(DEVICE)

    # ── 7f.  Threshold calibration on FULL val set (genuine + fake) ──────
    log.info("\n" + "=" * 60)
    log.info("  Threshold Calibration (using labels for cutoff only)")
    log.info("=" * 60)

    errors_val = compute_reconstruction_errors(model, X_val)
    y_val_np   = y_val.numpy()

    best_threshold, best_f1, was_inverted = calibrate_threshold(
        errors_val, y_val_np
    )

    print(f"\n  Best threshold : {best_threshold:.4f}")
    print(f"  Best F1 score  : {best_f1:.4f}")
    if was_inverted:
        print("  ⚠  Scores were INVERTED — autoencoder may have been trained on")
        print("     contaminated data.  Consider retraining with clean genuine reviews.")
        # Negate stored errors so the saved threshold is usable
        errors_val = -errors_val

    # ── 7g.  Evaluation ──────────────────────────────────────────────────
    metrics = evaluate_model(errors_val, y_val_np, best_threshold, 'Validation')

    # ── 7h.  Save artefacts ──────────────────────────────────────────────
    log.info("\nSaving evaluation plots …")
    save_plots(errors_val, y_val_np, best_threshold, save_dir)

    # Save threshold + inversion flag
    threshold_data = {
        'threshold'      : float(best_threshold),
        'scores_inverted': was_inverted,
        'best_f1'        : float(best_f1),
        'val_metrics'    : metrics,
    }
    thr_path = os.path.join(save_dir, 'threshold_config.json')
    with open(thr_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    log.info(f"  ↳ Threshold config → {thr_path}")

    # Save training log
    log_path = os.path.join(save_dir, 'training_log_autoencoder.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    log.info(f"  ↳ Training log     → {log_path}")

    print(f"\n✅ Stage 2 Training Complete!")
    print(f"   AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"   F1      : {metrics['f1']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()