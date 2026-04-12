"""
Evaluation utilities for ReviewGuard models.

Provides functions for:
- Metrics computation (accuracy, precision, recall, F1, ROC-AUC, etc.)
- Confusion matrix generation and visualization
- ROC curve plotting
- Cross-validation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, roc_auc_score, classification_report
)
from typing import Tuple, Dict
from pathlib import Path


class Evaluator:
    """
    Comprehensive model evaluation toolkit.
    """
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            y_probs: Predicted probabilities (for AUC-ROC)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_probs is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrix with heatmap.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6), dpi=150)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        title: str = "ROC Curve",
        save_path: str = None
    ) -> Tuple[float, float, float]:
        """
        Plot ROC curve.
        
        Args:
            y_true: Ground truth binary labels
            y_probs: Predicted probabilities
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(fpr, tpr, color='#6366F1', lw=2.5, 
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#D1D5DB', lw=2, linestyle='--',
                 label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.close()
        
        return fpr, tpr, roc_auc
    
    @staticmethod
    def plot_pr_curve(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: str = None
    ) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: Ground truth labels
            y_probs: Predicted probabilities
            title: Plot title
            save_path: Path to save
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(recall, precision, color='#10B981', lw=2.5,
                 label=f'PR Curve (AP = {ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PR curve saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list = None
    ) -> str:
        """
        Print detailed classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            target_names: Names for classes
            
        Returns:
            Report string
        """
        if target_names is None:
            target_names = ['Negative', 'Positive']
        
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4
        )
        
        print(report)
        return report
    
    @staticmethod
    def plot_training_curves(
        train_loss: list,
        val_loss: list,
        train_acc: list = None,
        val_acc: list = None,
        title: str = "Training History",
        save_path: str = None
    ) -> None:
        """
        Plot training and validation curves.
        
        Args:
            train_loss: Training loss per epoch
            val_loss: Validation loss per epoch
            train_acc: Training accuracy (optional)
            val_acc: Validation accuracy (optional)
            title: Plot title
            save_path: Path to save
        """
        epochs = range(1, len(train_loss) + 1)
        
        if train_acc and val_acc:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
            
            # Loss
            ax1.plot(epochs, train_loss, 'o-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Loss', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Accuracy
            ax2.plot(epochs, train_acc, 'o-', label='Training Acc', linewidth=2)
            ax2.plot(epochs, val_acc, 's-', label='Validation Acc', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Accuracy', fontsize=11)
            ax2.set_title('Accuracy', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.plot(epochs, train_loss, 'o-', label='Training Loss', linewidth=2)
            ax.plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title('Training History', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training curves saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def compute_bootstrap_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_iterations: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            n_iterations: Number of bootstrap samples
            
        Returns:
            Dictionary of {metric: (mean, std)}
        """
        n = len(y_true)
        metrics_bootstrap = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for _ in range(n_iterations):
            # Sample with replacement
            idx = np.random.choice(n, size=n, replace=True)
            y_t_boot = y_true[idx]
            y_p_boot = y_pred[idx]
            
            metrics_bootstrap['accuracy'].append(accuracy_score(y_t_boot, y_p_boot))
            metrics_bootstrap['precision'].append(
                precision_score(y_t_boot, y_p_boot, zero_division=0)
            )
            metrics_bootstrap['recall'].append(
                recall_score(y_t_boot, y_p_boot, zero_division=0)
            )
            metrics_bootstrap['f1'].append(f1_score(y_t_boot, y_p_boot, zero_division=0))
        
        results = {}
        for metric, values in metrics_bootstrap.items():
            results[metric] = (np.mean(values), np.std(values))
        
        return results
