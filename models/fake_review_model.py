"""
Stage 2: Supervised fake review detector optimized for small labeled datasets.

This model uses a hybrid word + character TF-IDF representation with a
linear classifier. It is a better fit than an unsupervised autoencoder when
only a few hundred labeled reviews are available.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


class FakeReviewDetector:
    """Hybrid TF-IDF fake review detector with calibrated probabilities."""

    def __init__(
        self,
        classifier_name: str = "linear_svm",
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_word_features: int = 12000,
        max_char_features: int = 18000,
        c_value: float = 1.0,
        random_state: int = 42,
    ):
        self.classifier_name = classifier_name
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.c_value = c_value
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.metadata: Dict = {}

    def _build_pipeline(self) -> Pipeline:
        features = FeatureUnion(
            transformer_list=[
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=True,
                        ngram_range=self.word_ngram_range,
                        min_df=self.min_df,
                        max_features=self.max_word_features,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        analyzer="char_wb",
                        sublinear_tf=True,
                        ngram_range=self.char_ngram_range,
                        min_df=self.min_df,
                        max_features=self.max_char_features,
                    ),
                ),
            ]
        )

        if self.classifier_name == "logistic_regression":
            classifier = LogisticRegression(
                C=self.c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=self.random_state,
                solver="liblinear",
            )
        elif self.classifier_name == "linear_svm":
            classifier = CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=self.c_value,
                    class_weight="balanced",
                    dual="auto",
                    random_state=self.random_state,
                ),
                cv=3,
            )
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_name}")

        return Pipeline(
            steps=[
                ("features", features),
                ("classifier", classifier),
            ]
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "FakeReviewDetector":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def predict_fake_probability(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    def predict(self, texts: Iterable[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

    def save_model(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as target:
            pickle.dump(
                {
                    "pipeline": self.pipeline,
                    "metadata": metadata or {},
                },
                target,
            )

    @staticmethod
    def load_model(filepath: str) -> "FakeReviewDetector":
        with open(filepath, "rb") as source:
            payload = pickle.load(source)

        detector = FakeReviewDetector()
        detector.pipeline = payload["pipeline"]
        detector.metadata = payload.get("metadata", {})
        return detector
