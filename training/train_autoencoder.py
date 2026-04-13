"""
Stage 2 Training: Supervised fake review detection.

This keeps the original command name for compatibility, but swaps the old
autoencoder for a stronger small-data detector based on TF-IDF features and a
linear classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from models.fake_review_model import FakeReviewDetector
from utils.preprocessing import TextPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"


CONFIG = {
    "data_mode": "real",
    "test_size": 0.20,
    "validation_size": 0.20,  # Fraction of the post-test training pool
    "min_samples_per_class": 40,
    "random_state": 42,
    "candidate_models": [
        {
            "name": "linear_svm_c1",
            "classifier_name": "linear_svm",
            "c_value": 1.0,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (3, 5),
        },
        {
            "name": "linear_svm_c2",
            "classifier_name": "linear_svm",
            "c_value": 2.0,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (3, 5),
        },
        {
            "name": "logreg_c2",
            "classifier_name": "logistic_regression",
            "c_value": 2.0,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (3, 5),
        },
    ],
}


def generate_synthetic_reviews(
    num_samples: int,
    review_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    genuine_phrases = [
        "This hotel was clean and comfortable, and the staff was consistently helpful.",
        "Excellent stay with smooth check-in and a quiet room.",
        "The location was convenient and the service felt genuine throughout.",
        "Comfortable bed, good breakfast, and overall a pleasant experience.",
        "Everything matched the listing and the staff handled requests quickly.",
    ]
    fake_phrases = [
        "BEST HOTEL EVER!!! You must book right now!!!",
        "Absolutely perfect in every way and changed my life forever!",
        "Everyone should stay here immediately, no downsides at all!",
        "Five stars for everything, unbelievable luxury, flawless service!!!",
        "This place is amazing amazing amazing and worth every penny forever!",
    ]
    phrases = genuine_phrases if review_type == "genuine" else fake_phrases
    return np.array([rng.choice(phrases) for _ in range(num_samples)])


def load_fake_review_data(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if config["data_mode"] == "real":
        data_path = DATA_DIR / "mexwell_reviews.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            df = df.dropna(subset=["review_text", "label"]).copy()
            df["review_text"] = df["review_text"].astype(str)
            df["label_id"] = (df["label"] == "YP").astype(int)
            return df["review_text"].values, df["label_id"].values

        print(f"⚠ Dataset not found at {data_path}")
        print("Falling back to synthetic data...")

    rng = np.random.default_rng(config["random_state"])
    genuine = generate_synthetic_reviews(600, "genuine", rng)
    fake = generate_synthetic_reviews(600, "fake", rng)
    reviews = np.concatenate([genuine, fake])
    labels = np.concatenate(
        [np.zeros(len(genuine), dtype=int), np.ones(len(fake), dtype=int)]
    )
    return reviews, labels


def split_dataset(reviews: np.ndarray, labels: np.ndarray, config: Dict):
    min_count = np.bincount(labels).min()
    if min_count < config["min_samples_per_class"]:
        raise ValueError(
            "Not enough samples per class for a stable supervised fake-review detector."
        )

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        reviews,
        labels,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=labels,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=config["validation_size"],
        random_state=config["random_state"],
        stratify=y_train_val,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob, best_threshold)
    best_key = (
        best_metrics["f1"],
        best_metrics["accuracy"],
        best_metrics["auc_roc"],
    )

    for threshold in np.linspace(0.30, 0.70, 81):
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        current_key = (
            metrics["f1"],
            metrics["accuracy"],
            metrics["auc_roc"],
        )
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = current_key

    return best_threshold, best_metrics


def evaluate_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
) -> Tuple[Dict, list[Dict]]:
    results = []
    best_result = None

    for candidate in config["candidate_models"]:
        start_time = time.time()
        detector = FakeReviewDetector(
            classifier_name=candidate["classifier_name"],
            c_value=candidate["c_value"],
            word_ngram_range=candidate["word_ngram_range"],
            char_ngram_range=candidate["char_ngram_range"],
            random_state=config["random_state"],
        )
        detector.fit(x_train, y_train)
        val_prob = detector.predict_proba(x_val)
        metrics = compute_metrics(y_val, val_prob, 0.5)
        elapsed = time.time() - start_time

        result = {
            "name": candidate["name"],
            "config": candidate,
            "metrics": metrics,
            "time_seconds": elapsed,
            "detector": detector,
        }
        results.append(result)

        if best_result is None:
            best_result = result
            continue

        current_key = (metrics["auc_roc"], metrics["f1"], metrics["accuracy"])
        best_key = (
            best_result["metrics"]["auc_roc"],
            best_result["metrics"]["f1"],
            best_result["metrics"]["accuracy"],
        )
        if current_key > best_key:
            best_result = result

    return best_result, results


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(12, 5), dpi=150)
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Genuine", color="#10B981")
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Fake", color="#F43F5E")
    plt.axvline(0.5, color="#6366F1", linestyle="--", linewidth=2, label="Threshold (0.5)")
    plt.xlabel("Predicted Fake Probability", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Stage 2 Probability Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, color="#6366F1", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#D1D5DB", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Fake Review Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("\n" + "=" * 60)
    print("🚀 ReviewGuard - Stage 2: Supervised Fake Review Detection")
    print("=" * 60)

    reviews, labels = load_fake_review_data(CONFIG)
    print(f"Loaded {len(reviews)} reviews")
    print(f"Genuine: {(labels == 0).sum()} | Fake: {(labels == 1).sum()}")

    preprocessor = TextPreprocessor()
    cleaned_reviews = np.array([preprocessor.clean_text(text) for text in reviews])

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        cleaned_reviews, labels, CONFIG
    )

    print("\nDataset split:")
    print(f"Train: {len(x_train)}")
    print(f"Val:   {len(x_val)}")
    print(f"Test:  {len(x_test)}")

    print("\nSelecting best Stage 2 model...")
    best_result, candidate_results = evaluate_candidates(
        x_train, y_train, x_val, y_val, CONFIG
    )

    for result in candidate_results:
        metrics = result["metrics"]
        print(
            f"{result['name']}: "
            f"Val AUC={metrics['auc_roc']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )

    print(f"\nBest validation model: {best_result['name']}")

    best_threshold, validation_metrics = find_best_threshold(
        y_val,
        best_result["detector"].predict_proba(x_val),
    )
    print(
        f"Best validation threshold: {best_threshold:.3f} "
        f"(F1={validation_metrics['f1']:.4f}, Acc={validation_metrics['accuracy']:.4f})"
    )

    final_detector = FakeReviewDetector(
        classifier_name=best_result["config"]["classifier_name"],
        c_value=best_result["config"]["c_value"],
        word_ngram_range=best_result["config"]["word_ngram_range"],
        char_ngram_range=best_result["config"]["char_ngram_range"],
        random_state=CONFIG["random_state"],
    )
    final_detector.fit(
        np.concatenate([x_train, x_val]),
        np.concatenate([y_train, y_val]),
    )

    test_prob = final_detector.predict_proba(x_test)
    metrics = compute_metrics(y_test, test_prob, best_threshold)

    print("\n" + "=" * 60)
    print("📈 Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fake_review_detector.pkl"
    metadata = {
        "threshold": best_threshold,
        "config": CONFIG,
        "selected_model": best_result["config"],
        "validation_metrics": validation_metrics,
        "metrics": metrics,
    }
    final_detector.save_model(str(model_path), metadata=metadata)
    print(f"✓ Detector saved to {model_path}")

    plot_probability_distribution(
        y_test,
        test_prob,
        MODEL_DIR / "reconstruction_error_distribution.png",
    )
    plot_roc_curve(y_test, test_prob, MODEL_DIR / "roc_curve_autoencoder.png")
    print("✓ Evaluation plots saved")

    log_path = MODEL_DIR / "training_log_autoencoder.json"
    with open(log_path, "w", encoding="utf-8") as target:
        json.dump(
            {
                "stage2_model_type": "supervised_text_detector",
                "threshold": best_threshold,
                "candidate_results": [
                    {
                        "name": result["name"],
                        "config": result["config"],
                        "metrics": result["metrics"],
                        "time_seconds": result["time_seconds"],
                    }
                    for result in candidate_results
                ],
                "selected_model": best_result["config"],
                "validation_metrics": validation_metrics,
                "metrics": metrics,
                "split_sizes": {
                    "train": int(len(x_train)),
                    "val": int(len(x_val)),
                    "test": int(len(x_test)),
                },
            },
            target,
            indent=2,
        )
    print(f"✓ Logs saved to {log_path}")
    print("\n✅ Stage 2 Training Complete!\n")


if __name__ == "__main__":
    main()
