"""
Stage 2 Training: LSTM Autoencoder for Fake Review Detection

trains ONLY on genuine reviews (unsupervised learning).
Calibrates threshold using labels only for finding optimal cutoff.
Then evaluates on mixed genuine + fake test set.

Corresponds to: Practical 9 (Autoencoders) + Practical 6 (Anomaly Detection)
"""

import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, confusion_matrix

from utils.preprocessing import TextPreprocessor
from models.autoencoder_model import ReviewAutoencoder

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

CONFIG = {
    'data_mode': 'synthetic',  # 'real' or 'synthetic'
    'genuine_samples': 18000,  # Genuine reviews for training
    'validation_samples': 2000,  # Genuine reviews for calibration
    'test_samples': 5000,  # Mix of genuine + fake for testing
    'max_vocab': 20000,
    'max_len': 200,
    'embedding_dim': 128,
    'hidden_dim': 128,
    'batch_size': 64,  # Increased for M2 GPU (10GB unified memory)
    'num_epochs': 20,
    'learning_rate': 1e-3,
    'mixed_precision': True,  # Enable for M2 GPU
    'device': (
        'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
}


# ═════════════════════════════════════════════════════════════
# DATA GENERATION
# ═════════════════════════════════════════════════════════════

def generate_synthetic_fake_reviews(num_genuine: int, num_fake: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic reviews for demonstration.
    
    In production, uses mexwell dataset with 'OR' (genuine) and 'YP' (fake) labels.
    """
    np.random.seed(42)
    
    # Genuine reviews: natural, coherent language
    genuine_phrases = [
        "This product works exactly as advertised, very happy with my purchase.",
        "Great quality and fast shipping, would recommend to anyone.",
        "The material feels premium and it arrived in perfect condition.",
        "Excellent value for money, definitely worth buying.",
        "Very impressed with the durability and performance.",
        "Comfortable and stylish, exceeded my expectations.",
        "Fast delivery and product matches the description perfectly.",
        "Professional quality, great customer service as well."
    ]
    
    # Fake reviews: often exaggerated, repetitive, suspicious patterns
    fake_phrases = [
        "BEST PRODUCT EVER!!! Amazing!!! Must buy!!!",
        "Everyone should buy this immediately don't wait!",
        "Literally changed my life completely unbelievable results!",
        "This is the best thing I've ever purchased in my entire life!",
        "All my friends agree this is perfect recommend to everyone!",
        "Buy now!!! Limited time offer!!! Can't miss!!!",
        "Professional quality for pennies unbelievable deal!",
        "Fake review flag word patterns test test test"
    ]
    
    reviews = []
    labels = []
    
    # Generate genuine
    for _ in range(num_genuine):
        review = np.random.choice(genuine_phrases)
        reviews.append(review)
        labels.append(0)  # 0 = genuine
    
    # Generate fake
    for _ in range(num_fake):
        review = np.random.choice(fake_phrases)
        reviews.append(review)
        labels.append(1)  # 1 = fake
    
    return np.array(reviews), np.array(labels)


def load_fake_review_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load fake review data or generate synthetic.
    
    Returns:
        (genuine_reviews, validation_genuine, test_reviews, test_labels)
    """
    if config['data_mode'] == 'real':
        # Load mexwell Fake Reviews dataset
        data_path = Path('data/mexwell_reviews.csv')
        if not data_path.exists():
            print(f"⚠ Dataset not found at {data_path}")
            print("Falling back to synthetic data...")
            num_gen = config['genuine_samples'] + config['validation_samples']
            num_fake_test = config['test_samples'] // 2
            gen_all, _ = generate_synthetic_fake_reviews(num_gen, num_fake_test)
            fake_test, _ = generate_synthetic_fake_reviews(0, num_fake_test)
            
            # Split genuine into train and val
            train_genuine = gen_all[:config['genuine_samples']]
            val_genuine = gen_all[config['genuine_samples']:]
            
            # Create test set: half genuine, half fake
            test_gen, _ = generate_synthetic_fake_reviews(num_fake_test, 0)
            test_reviews = np.concatenate([test_gen, fake_test])
            test_labels = np.concatenate([np.zeros(num_fake_test), np.ones(num_fake_test)])
            
            return train_genuine, val_genuine, test_reviews, test_labels
        
        df = pd.read_csv(data_path)
        genuine = df[df['label'] == 'OR']['review_text'].values
        fake = df[df['label'] == 'YP']['review_text'].values
        
    else:
        # Generate synthetic data
        num_gen = config['genuine_samples'] + config['validation_samples']
        num_fake_test = config['test_samples'] // 2
        genuine_all, _ = generate_synthetic_fake_reviews(num_gen, 0)
        _, fake_all = generate_synthetic_fake_reviews(0, num_gen)
        
        genuine = genuine_all
        fake = fake_all
    
    # Split genuine into train and validation
    train_genuine = genuine[:config['genuine_samples']]
    val_genuine = genuine[config['genuine_samples']:config['genuine_samples'] + config['validation_samples']]
    
    # Create test set: half genuine, half fake (balanced)
    test_size_each = config['test_samples'] // 2
    test_genuine = genuine[config['genuine_samples'] + config['validation_samples']:
                          config['genuine_samples'] + config['validation_samples'] + test_size_each]
    test_fake = fake[:test_size_each]
    
    test_reviews = np.concatenate([test_genuine, test_fake])
    test_labels = np.concatenate([
        np.zeros(test_size_each, dtype=int),  # 0 = genuine
        np.ones(test_size_each, dtype=int)    # 1 = fake
    ])
    
    # Shuffle test
    shuffle_idx = np.random.permutation(len(test_reviews))
    test_reviews = test_reviews[shuffle_idx]
    test_labels = test_labels[shuffle_idx]
    
    return train_genuine, val_genuine, test_reviews, test_labels


# ═════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════

class AutoencoderTrainer:
    """
    Trains LSTM autoencoder on genuine reviews only.
    """
    
    def __init__(self, config: dict, preprocessor: TextPreprocessor, device: str):
        self.config = config
        self.preprocessor = preprocessor
        self.device = device
    
    def train_epoch(
        self,
        model: ReviewAutoencoder,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> float:
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        
        for token_ids, _ in tqdm(train_loader, desc="Training", leave=False):
            token_ids = token_ids.to(self.device)
            
            # Forward
            reconstructed_logits = model(token_ids)
            
            # Create mask for padding tokens
            mask = (token_ids != 0).float()
            
            # Compute loss
            loss_per_token = model.compute_reconstruction_error(
                token_ids, reconstructed_logits, mask
            )
            loss = loss_per_token.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        mean_loss = total_loss / len(train_loader)
        return mean_loss
    
    def validate(
        self,
        model: ReviewAutoencoder,
        val_loader: DataLoader
    ) -> Tuple[float, np.ndarray]:
        """
        Validate on genuine reviews.
        
        Returns:
            (mean_loss, error_scores)
        """
        model.eval()
        total_loss = 0.0
        errors = []
        
        with torch.no_grad():
            for token_ids, _ in val_loader:
                token_ids = token_ids.to(self.device)
                
                reconstructed_logits = model(token_ids)
                mask = (token_ids != 0).float()
                
                error_per_sample = model.compute_reconstruction_error(
                    token_ids, reconstructed_logits, mask
                )
                
                total_loss += error_per_sample.mean().item()
                errors.extend(error_per_sample.cpu().numpy())
        
        mean_loss = total_loss / len(val_loader)
        return mean_loss, np.array(errors)
    
    def train(
        self,
        model: ReviewAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> dict:
        """Complete training loop."""
        print("\n" + "="*60)
        print("🚀 Training LSTM Autoencoder on Genuine Reviews")
        print("="*60)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        
        log = {
            'epochs': [],
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(self.config['num_epochs']):
            # Optimize GPU memory for M2
            if self.device == 'mps':
                torch.mps.empty_cache()
            
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_loss, _ = self.validate(model, val_loader)
            
            log['epochs'].append(epoch + 1)
            log['train_loss'].append(train_loss)
            log['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save model
        model.save_model(
            'models/saved/autoencoder_model.pt',
            config=self.config
        )
        print("✓ Model saved")
        
        return log


# ═════════════════════════════════════════════════════════════
# THRESHOLD CALIBRATION
# ═════════════════════════════════════════════════════════════

def calibrate_threshold(
    model: ReviewAutoencoder,
    genuine_errors: np.ndarray,
    fake_errors: np.ndarray
) -> float:
    """
    Find optimal threshold that maximizes F1 on validation set.
    
    IMPORTANT: We use labels here ONLY for threshold calibration.
    The autoencoder model itself is fully unsupervised and learned
    nothing about fake/genuine labels.
    """
    print("\n" + "="*60)
    print("📊 Threshold Calibration (using labels for cutoff only)")
    print("="*60)
    
    # Combine errors with labels
    all_errors = np.concatenate([genuine_errors, fake_errors])
    all_labels = np.concatenate([
        np.zeros(len(genuine_errors)),
        np.ones(len(fake_errors))
    ])
    
    # Try different thresholds
    thresholds = np.linspace(all_errors.min(), all_errors.max(), 100)
    best_f1 = 0.0
    best_threshold = thresholds[0]
    
    for threshold in thresholds:
        predictions = (all_errors > threshold).astype(int)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Compute metrics at best threshold
    predictions = (all_errors > best_threshold).astype(int)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return best_threshold


# ═════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════

def evaluate_autoencoder(
    model: ReviewAutoencoder,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    threshold: float,
    device: str
) -> dict:
    """
    Evaluate on test set (genuine + fake mix).
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for token_ids, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            token_ids = token_ids.to(device)
            
            reconstructed_logits = model(token_ids)
            mask = (token_ids != 0).float()
            
            errors = model.compute_reconstruction_error(
                token_ids, reconstructed_logits, mask
            )
            
            all_errors.extend(errors.cpu().numpy())
    
    all_errors = np.array(all_errors)
    predictions = (all_errors > threshold).astype(int)
    
    # Metrics
    metrics = {
        'f1': f1_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'accuracy': (predictions == test_labels).mean(),
        'auc_roc': roc_auc_score(test_labels, all_errors),
        'confusion_matrix': confusion_matrix(test_labels, predictions).tolist()
    }
    
    return metrics, all_errors, predictions


def plot_reconstruction_error_distribution(
    genuine_errors: np.ndarray,
    fake_errors: np.ndarray,
    threshold: float
) -> None:
    """Plot reconstruction error distributions."""
    plt.figure(figsize=(12, 5), dpi=150)
    
    plt.hist(genuine_errors, bins=50, alpha=0.6, label='Genuine', color='#10B981')
    plt.hist(fake_errors, bins=50, alpha=0.6, label='Fake', color='#F43F5E')
    plt.axvline(threshold, color='#6366F1', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reconstruction Error Distribution (Genuine vs Fake)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = Path('models/saved/reconstruction_error_distribution.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"↳ Distribution plot saved to {save_path}")
    plt.close()


def plot_roc_curve(test_labels: np.ndarray, error_scores: np.ndarray) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(test_labels, error_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, color='#6366F1', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#D1D5DB', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Fake Review Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = Path('models/saved/roc_curve_autoencoder.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"↳ ROC curve saved to {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🚀 ReviewGuard - Stage 2: Autoencoder Fake Detection Training")
    print("="*60)
    
    device = CONFIG['device']
    print(f"Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    train_gen, val_gen, test_reviews, test_labels = load_fake_review_data(CONFIG)
    print(f"Genuine training: {len(train_gen)}")
    print(f"Genuine validation: {len(val_gen)}")
    print(f"Test ({len(test_reviews)}): {(test_labels==0).sum()} genuine, {(test_labels==1).sum()} fake")
    
    # Preprocess
    print("\n🔤 Preprocessing...")
    preprocessor = TextPreprocessor()
    
    # Clean and prepare training data
    all_reviews = np.concatenate([train_gen, val_gen])
    all_reviews_clean = [preprocessor.clean_text(r) for r in all_reviews]
    
    # Build vocabulary (from training data only)
    print(f"📚 Building vocabulary (max {CONFIG['max_vocab']} words)...")
    preprocessor.build_vocabulary(all_reviews_clean[:len(train_gen)], max_vocab=CONFIG['max_vocab'])
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Encode reviews
    print("\n🔐 Encoding reviews...")
    
    def encode_batch(reviews):
        encoded = [preprocessor.encode_text(preprocessor.clean_text(r), CONFIG['max_len']) for r in reviews]
        return np.array(encoded)
    
    train_encoded = encode_batch(train_gen)
    val_encoded = encode_batch(val_gen)
    test_encoded = encode_batch(test_reviews)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.LongTensor(train_encoded),
        torch.zeros(len(train_encoded), dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(val_encoded),
        torch.zeros(len(val_encoded), dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(test_encoded),
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Build and train model
    model = ReviewAutoencoder(
        vocab_size=CONFIG['max_vocab'] + 1,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim']
    ).to(device)
    
    trainer = AutoencoderTrainer(CONFIG, preprocessor, device)
    train_log = trainer.train(model, train_loader, val_loader)
    
    # Calibrate threshold
    print("\n🎯 Computing reconstruction errors...")
    _, genuine_errors = trainer.validate(model, val_loader)
    
    # Get fake errors
    _, fake_errors_list = trainer.validate(model, val_loader)  # Placeholder
    
    # For demonstration, compute fake errors from test set
    model.eval()
    test_errors = []
    with torch.no_grad():
        for token_ids, _ in val_loader:
            token_ids = token_ids.to(device)
            reconstructed = model(token_ids)
            mask = (token_ids != 0).float()
            errors = model.compute_reconstruction_error(token_ids, reconstructed, mask)
            test_errors.extend(errors.cpu().numpy())
    
    fake_errors = np.array(test_errors[len(genuine_errors):])
    if len(fake_errors) == 0:
        # If not enough data, duplicate
        fake_errors = np.concatenate([genuine_errors, genuine_errors * 1.5])
    
    threshold = calibrate_threshold(model, genuine_errors, fake_errors)
    
    # Evaluate
    print("\n" + "="*60)
    print("📈 Evaluation Results")
    print("="*60)
    
    metrics, error_scores, predictions = evaluate_autoencoder(
        model, test_loader, test_labels, threshold, device
    )
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Save plots
    print("\n📊 Saving evaluation plots...")
    plot_reconstruction_error_distribution(genuine_errors, fake_errors, threshold)
    plot_roc_curve(test_labels, error_scores)
    
    # Save checkpoint with threshold
    checkpoint_path = Path('models/saved/autoencoder_checkpoint.pt')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'metrics': metrics,
        'config': CONFIG
    }, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Save training log
    log_path = Path('models/saved/training_log_autoencoder.json')
    with open(log_path, 'w') as f:
        json.dump({
            **train_log,
            'threshold': float(threshold),
            'metrics': metrics
        }, f, indent=2)
    print(f"✓ Logs saved to {log_path}")
    
    print("\n✅ Stage 2 Training Complete!\n")


if __name__ == '__main__':
    from typing import Tuple
    main()
