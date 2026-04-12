"""
Stage 1 Training: LSTM Sentiment Classifier

Trains on Amazon Reviews (bittlingmayer dataset).
Compares 3 optimizers (Adam, SGD, RMSProp) and regularization techniques.
Produces training logs and confusion matrix for analysis.

Corresponds to: Practical 7 (Optimization comparison) + Practical 5 (Regularization)
"""

import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing import TextPreprocessor
from models.sentiment_model import SentimentLSTM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models' / 'saved'

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

CONFIG = {
    'data_mode': 'real',  # Falls back to synthetic if the dataset is unavailable
    'max_samples': 200000,  # 100K positive + 100K negative
    'max_vocab': 20000,
    'max_len': 200,  # 95th percentile of review lengths
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.2,  # Reduced from 0.3 to prevent divergence
    'batch_size': 64,  # Increased for M2 GPU (10GB unified memory)
    'num_epochs': 10,
    'learning_rate': {
        'adam': 5e-4,  # Reduced from 1e-3 (was too aggressive)
        'sgd': 0.005,  # Reduced from 0.01
        'rmsprop': 5e-4  # Reduced from 1e-3
    },
    'regularization': {
        'l2_weight_decay': 1e-4,
        'l1_factor': 0.0,
        'early_stopping_patience': 5,  # Increased from 3 to allow recovery
        'gradient_clip': 1.0,  # Prevent gradient explosion
        'warmup_epochs': 1  # Stabilize first epoch
    },
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

def generate_synthetic_sentiment_data(num_samples: int = 200000) -> pd.DataFrame:
    """
    Generate synthetic sentiment data for demonstration.
    
    In production, this would load real Amazon Reviews data.
    """
    np.random.seed(42)
    
    positive_words = [
        'amazing', 'excellent', 'great', 'awesome', 'perfect', 'love',
        'wonderful', 'fantastic', 'brilliant', 'outstanding', 'superb',
        'incredible', 'impressed', 'satisfied', 'recommended', 'quality'
    ]
    
    negative_words = [
        'terrible', 'awful', 'bad', 'horrible', 'waste', 'poor', 'disappointing',
        'broken', 'defective', 'useless', 'disaster', 'regret', 'avoid',
        'overpriced', 'cheap', 'refund'
    ]
    
    phrases = [
        'This product is {}.',
        'I {} this {}!',
        'Really {} experience with {}.',
        'The {} is absolutely {}.',
        'Would {} recommend!'
    ]
    
    reviews = []
    ratings = []
    
    # Generate positive reviews
    for _ in range(num_samples // 2):
        words = np.random.choice(positive_words, size=np.random.randint(5, 20))
        review = ' '.join(words)
        rating = np.random.choice([4, 5])
        reviews.append(review)
        ratings.append(rating)
    
    # Generate negative reviews
    for _ in range(num_samples // 2):
        words = np.random.choice(negative_words, size=np.random.randint(5, 20))
        review = ' '.join(words)
        rating = np.random.choice([1, 2])
        reviews.append(review)
        ratings.append(rating)
    
    df = pd.DataFrame({'rating': ratings, 'review': reviews})
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def load_sentiment_data(config: dict) -> pd.DataFrame:
    """
    Load sentiment data from Amazon Reviews or generate synthetic.
    """
    if config['data_mode'] == 'real':
        # Load real Amazon Reviews
        data_path = DATA_DIR / 'amazon_reviews_train.csv'
        if not data_path.exists():
            print(f"⚠ Dataset not found at {data_path}")
            print("Falling back to synthetic data for demonstration...")
            return generate_synthetic_sentiment_data(config['max_samples'])
        
        df = pd.read_csv(data_path, header=None, names=['rating', 'review'])
        # Sample to max_samples
        if len(df) > config['max_samples']:
            df = df.sample(n=config['max_samples'], random_state=42)
        return df
    
    else:
        # Generate synthetic data
        return generate_synthetic_sentiment_data(config['max_samples'])


# ═════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════

class SentimentTrainer:
    """
    Trains sentiment classifier with configurable optimizer and regularization.
    """
    
    def __init__(self, config: dict, preprocessor: TextPreprocessor, device: str):
        """
        Args:
            config: Configuration dictionary
            preprocessor: TextPreprocessor with built vocabulary
            device: 'cuda' or 'cpu'
        """
        self.config = config
        self.preprocessor = preprocessor
        self.device = device
        self.training_log = {}
    
    def train_epoch(
        self,
        model: SentimentLSTM,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        l1_factor: float = 0.0
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            model: Sentiment model
            train_loader: Training DataLoader
            loss_fn: Binary cross-entropy loss
            optimizer: PyTorch optimizer
            l1_factor: L1 regularization factor
            
        Returns:
            Tuple of (mean loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for token_ids, labels in tqdm(train_loader, desc="Training", leave=False):
            token_ids = token_ids.to(self.device)
            labels = labels.to(self.device).float()
            
            # Forward pass
            logits, _ = model.forward_logits(token_ids)
            predictions = torch.sigmoid(logits)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            
            # Add L1 regularization if specified
            if l1_factor > 0:
                l1_penalty = 0.0
                for param in model.parameters():
                    l1_penalty += torch.abs(param).sum()
                loss = loss + l1_factor * l1_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.config['regularization']['gradient_clip']
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend((predictions > 0.5).cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        mean_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return mean_loss, accuracy
    
    def validate(
        self,
        model: SentimentLSTM,
        val_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model on validation set.
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for token_ids, labels in val_loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device).float()
                
                logits, _ = model.forward_logits(token_ids)
                predictions = torch.sigmoid(logits)
                loss = loss_fn(logits, labels)
                
                total_loss += loss.item()
                all_preds.extend((predictions > 0.5).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        mean_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return mean_loss, accuracy
    
    def train(
        self,
        model: SentimentLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_name: str = 'adam'
    ) -> dict:
        """
        Complete training loop with early stopping.
        
        Args:
            model: Sentiment model to train
            train_loader: Training data
            val_loader: Validation data
            optimizer_name: 'adam', 'sgd', or 'rmsprop'
            
        Returns:
            Training log dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training with {optimizer_name.upper()}")
        print(f"{'='*60}")
        
        # Setup optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate']['adam'],
                weight_decay=self.config['regularization']['l2_weight_decay']
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate']['sgd'],
                momentum=0.9,
                weight_decay=self.config['regularization']['l2_weight_decay']
            )
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config['learning_rate']['rmsprop'],
                weight_decay=self.config['regularization']['l2_weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Early stopping
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = self.config['regularization']['early_stopping_patience']
        
        log = {
            'optimizer': optimizer_name,
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'times': []
        }
        
        for epoch in range(self.config['num_epochs']):
            # Optimize GPU memory for M2
            if self.device == 'mps':
                torch.mps.empty_cache()
            
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, loss_fn, optimizer,
                l1_factor=self.config['regularization']['l1_factor']
            )
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, loss_fn)
            
            epoch_time = time.time() - epoch_start
            
            log['epochs'].append(epoch + 1)
            log['train_loss'].append(train_loss)
            log['val_loss'].append(val_loss)
            log['train_acc'].append(train_acc)
            log['val_acc'].append(val_acc)
            log['times'].append(epoch_time)
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                model.save_model(
                    str(MODEL_DIR / f'sentiment_model_{optimizer_name}_best.pt'),
                    config=self.config,
                    best_val_accuracy=best_val_accuracy
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        log['best_val_accuracy'] = best_val_accuracy
        return log


# ═════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════

def evaluate_model(
    model: SentimentLSTM,
    test_loader: DataLoader,
    device: str
) -> dict:
    """
    Evaluate model on test set with detailed metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for token_ids, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            token_ids = token_ids.to(device)
            predictions, _ = model(token_ids)
            
            all_probs.extend(predictions.cpu().numpy())
            all_preds.extend((predictions > 0.5).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    return metrics, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, title: str = "Sentiment Classification") -> None:
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save
    save_path = MODEL_DIR / 'confusion_matrix_sentiment.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"↳ Confusion matrix saved to {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*60)
    print("🚀 ReviewGuard - Stage 1: Sentiment Classification Training")
    print("="*60)
    
    device = CONFIG['device']
    print(f"Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    df = load_sentiment_data(CONFIG)
    print(f"Loaded {len(df)} reviews")
    
    # Preprocess
    print("\n🔤 Preprocessing text...")
    preprocessor = TextPreprocessor()
    
    # Clean texts
    df['review_clean'] = df['review'].apply(preprocessor.clean_text)
    
    # Map labels (skip 3-star reviews)
    df['label'] = df['rating'].apply(preprocessor.map_sentiment_label)
    df = df[df['label'].notna()].reset_index(drop=True)
    print(f"After filtering: {len(df)} reviews ({df['label'].sum()} positive, {(1-df['label']).sum()} negative)")
    
    # Build vocabulary
    print(f"\n📚 Building vocabulary (max {CONFIG['max_vocab']} words)...")
    word2idx, idx2word = preprocessor.build_vocabulary(
        df['review_clean'].tolist(),
        max_vocab=CONFIG['max_vocab']
    )
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Save vocabulary
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = MODEL_DIR / 'vocabulary.pkl'
    preprocessor.save_vocabulary(str(vocab_path))
    print(f"✓ Vocabulary saved")
    
    # Encode reviews
    print("\n🔐 Encoding reviews...")
    encoded_reviews = []
    for review in tqdm(df['review_clean'], desc="Encoding", leave=False):
        encoded = preprocessor.encode_text(review, max_len=CONFIG['max_len'])
        encoded_reviews.append(encoded)
    
    X = np.array(encoded_reviews)
    y = df['label'].values.astype(np.int64)
    
    print(f"Data shape: {X.shape}")
    
    # Split data 80/10/10 with stratification so class balance remains stable.
    print("\n✂️ Splitting data (80/10/10, stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_data = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    test_data = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'])
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train models with different optimizers
    print("\n" + "="*60)
    print("🤖 Training models with different optimizers")
    print("="*60)
    
    trainer = SentimentTrainer(CONFIG, preprocessor, device)
    all_logs = {}
    best_model_path = None
    best_accuracy = 0.0
    
    for optimizer_name in ['adam', 'sgd', 'rmsprop']:
        model = SentimentLSTM(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        ).to(device)
        
        log = trainer.train(model, train_loader, val_loader, optimizer_name)
        all_logs[optimizer_name] = log
        
        if log['best_val_accuracy'] > best_accuracy:
            best_accuracy = log['best_val_accuracy']
            best_model_path = str(MODEL_DIR / f'sentiment_model_{optimizer_name}_best.pt')
    
    # Evaluate best model
    print("\n" + "="*60)
    print("📈 Evaluation Results")
    print("="*60)
    
    best_model = SentimentLSTM.load_model(best_model_path, device)
    metrics, all_preds, all_labels = evaluate_model(best_model, test_loader, device)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # Save confusion matrix
    plot_confusion_matrix(all_labels, all_preds)
    
    # Save training log
    print("\n💾 Saving training logs...")
    log_path = MODEL_DIR / 'training_log_sentiment.json'
    with open(log_path, 'w') as f:
        # Convert to serializable format
        serializable_logs = {}
        for opt, log in all_logs.items():
            serializable_logs[opt] = {
                k: v if not isinstance(v, list) else list(v)
                for k, v in log.items()
            }
        serializable_logs['best_model'] = {
            'optimizer': best_model_path.split('_')[2].replace('.pt', ''),
            'metrics': metrics
        }
        json.dump(serializable_logs, f, indent=2)
    print(f"✓ Logs saved to {log_path}")
    
    print("\n✅ Stage 1 Training Complete!\n")


if __name__ == '__main__':
    main()
