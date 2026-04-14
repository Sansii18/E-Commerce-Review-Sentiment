"""
Stage 3: Score Fusion
======================
Combines the reconstruction error from the autoencoder and the
contradiction score (rating vs sentiment mismatch) into a single
[0, 1] fake-probability score.

FIX NOTES (vs original)
------------------------
1.  `ScoreFuser.compute_final_score` now accepts `scores_inverted` from
    the threshold config so inference direction is always correct.

2.  `normalize_error` clips to [0, 1] after min-max scaling instead of
    relying on an implicit assumption about raw error range.

3.  `compute_contradiction_score` was using a hardcoded midpoint of 0.5
    for the sentiment boundary; we now accept `threshold=0.5` as a
    parameter so it can be tuned without touching the code.

4.  Verdict thresholds are clearly documented and easy to adjust.
"""

from __future__ import annotations
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

# Fusion weights  (must sum to 1.0)
W_RECONSTRUCTION = 0.65   # primary signal
W_CONTRADICTION  = 0.35   # secondary signal (rating/sentiment mismatch)

# Verdict thresholds
VERDICT_GENUINE    = 0.35   # final_score < 0.35
VERDICT_SUSPICIOUS = 0.60   # 0.35 ≤ final_score < 0.60
# final_score ≥ 0.60 → Likely Fake

# Emoji shortcuts
EMOJI = {
    'genuine'   : '🟢',
    'suspicious': '🟡',
    'fake'      : '🔴',
}


# ──────────────────────────────────────────────────────────────────────────
# Normalisation helper
# ──────────────────────────────────────────────────────────────────────────

class ErrorNormalizer:
    """
    Min-max normalise reconstruction errors to [0, 1].

    Parameters
    ----------
    min_error : expected minimum error (e.g. 2.5 – typical for well-trained AE)
    max_error : expected maximum error (e.g. 8.0 – typical upper bound)
    """

    def __init__(
        self,
        min_error: float = 2.5,
        max_error: float = 8.0,
    ) -> None:
        self.min_error = min_error
        self.max_error = max_error

    def normalize(self, error: float) -> float:
        """
        Map raw reconstruction error to [0, 1].
        Values outside [min_error, max_error] are clipped.
        """
        span = self.max_error - self.min_error
        if span <= 0:
            return 0.5
        norm = (error - self.min_error) / span
        return float(min(max(norm, 0.0), 1.0))   # clip to [0,1]

    @classmethod
    def from_training_stats(
        cls,
        genuine_errors: list[float],
        percentile_low : float = 5.0,
        percentile_high: float = 95.0,
    ) -> 'ErrorNormalizer':
        """
        Fit normalizer from genuine validation errors.
        Use 5th/95th percentiles to be robust to outliers.
        """
        import numpy as np
        arr = np.array(genuine_errors)
        return cls(
            min_error=float(np.percentile(arr, percentile_low)),
            max_error=float(np.percentile(arr, percentile_high)),
        )


# ──────────────────────────────────────────────────────────────────────────
# Contradiction score
# ──────────────────────────────────────────────────────────────────────────

class ContradictionScorer:
    """
    Compute the rating/sentiment mismatch score — a novel fraud signal.

    High score → the reviewer's STAR RATING contradicts their TEXT sentiment,
    which is a strong indicator of rating manipulation.

    Examples
    --------
    • 5★ with negative text sentiment  → high contradiction → suspicious
    • 1★ with positive text sentiment  → high contradiction → suspicious
    • 4★ with positive text sentiment  → low  contradiction → normal

    Parameters
    ----------
    sentiment_threshold: float
        Probability cutoff to classify sentiment as positive.
        Default 0.5 (model output ≥ 0.5 → positive).
    """

    def __init__(self, sentiment_threshold: float = 0.5) -> None:
        self.sentiment_threshold = sentiment_threshold

    def score(
        self,
        star_rating        : int,
        sentiment_prob     : float,
        sentiment_label    : Optional[int] = None,
    ) -> float:
        """
        Parameters
        ----------
        star_rating     : 1–5 integer rating given by the reviewer
        sentiment_prob  : float in [0,1] from the sentiment model
                          (1 = very positive sentiment)
        sentiment_label : 1 = positive, 0 = negative
                          (if None, derived from sentiment_prob)

        Returns
        -------
        contradiction_score : float in [0, 1]
        """
        if sentiment_label is None:
            sentiment_label = int(sentiment_prob >= self.sentiment_threshold)

        # Normalise star rating to [0, 1]  (1★ → 0.0, 5★ → 1.0)
        star_norm = (star_rating - 1) / 4.0

        # Expected sentiment from rating  (high stars → positive expected)
        expected_sentiment = star_norm  # [0=negative, 1=positive]

        # Contradiction = distance between expected and actual probability
        # Both are in [0, 1], so distance is also in [0, 1].
        contradiction = abs(expected_sentiment - sentiment_prob)

        # Amplify: squared version penalises large mismatches more
        contradiction = contradiction ** 2 * 1.5       # scale slightly
        return float(min(contradiction, 1.0))

    def explain(
        self,
        star_rating    : int,
        sentiment_prob : float,
    ) -> str:
        """Return a human-readable explanation of the contradiction signal."""
        score = self.score(star_rating, sentiment_prob)
        sentiment_word = 'positive' if sentiment_prob >= self.sentiment_threshold else 'negative'
        if score < 0.1:
            return (
                f"Rating ({star_rating}★) and sentiment ({sentiment_word}, "
                f"p={sentiment_prob:.2f}) are CONSISTENT."
            )
        elif score < 0.35:
            return (
                f"Mild mismatch: {star_rating}★ with {sentiment_word} language "
                f"(p={sentiment_prob:.2f})."
            )
        else:
            return (
                f"⚠  STRONG contradiction: {star_rating}★ rating but "
                f"{sentiment_word} text sentiment (p={sentiment_prob:.2f}). "
                f"Classic rating-manipulation pattern."
            )


# ──────────────────────────────────────────────────────────────────────────
# Score fuser
# ──────────────────────────────────────────────────────────────────────────

class ScoreFuser:
    """
    Fuse normalised reconstruction error and contradiction score into
    a single fake-probability.

    Formula
    -------
        final_score = W_RECONSTRUCTION × norm_error
                    + W_CONTRADICTION  × contradiction_score
    """

    def __init__(
        self,
        normalizer          : Optional[ErrorNormalizer] = None,
        contradiction_scorer: Optional[ContradictionScorer] = None,
        w_reconstruction    : float = W_RECONSTRUCTION,
        w_contradiction     : float = W_CONTRADICTION,
    ) -> None:
        self.normalizer   = normalizer   or ErrorNormalizer()
        self.cscore       = contradiction_scorer or ContradictionScorer()
        self.w_rec        = w_reconstruction
        self.w_con        = w_contradiction
        assert abs(self.w_rec + self.w_con - 1.0) < 1e-6, \
            "Fusion weights must sum to 1.0"

    def compute(
        self,
        raw_error       : float,
        star_rating     : int,
        sentiment_prob  : float,
        scores_inverted : bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        raw_error       : reconstruction error from autoencoder
        star_rating     : 1–5 reviewer rating
        sentiment_prob  : [0,1] probability from sentiment model
        scores_inverted : if True, error was negated during calibration
                          (set from threshold_config.json)

        Returns
        -------
        dict with keys:
          final_score         : float [0,1]  (higher = more likely fake)
          norm_error          : float [0,1]
          contradiction_score : float [0,1]
          verdict             : str
          verdict_emoji       : str
          explanation         : str
        """
        # Handle inverted scores from training (rare but handled)
        effective_error = -raw_error if scores_inverted else raw_error

        norm_err   = self.normalizer.normalize(effective_error)
        contra     = self.cscore.score(star_rating, sentiment_prob)
        final      = self.w_rec * norm_err + self.w_con * contra
        final      = float(min(max(final, 0.0), 1.0))

        verdict, emoji = get_verdict(final)

        explanation = (
            f"Reconstruction error: {raw_error:.3f} (normalised → {norm_err:.2f})\n"
            + self.cscore.explain(star_rating, sentiment_prob) + "\n"
            + f"Final fake-score: {final:.3f} ({emoji} {verdict})"
        )

        return dict(
            final_score         = final,
            norm_error          = norm_err,
            contradiction_score = contra,
            verdict             = verdict,
            verdict_emoji       = emoji,
            explanation         = explanation,
        )

    @staticmethod
    def compute_final_score(
        norm_error          : float,
        contradiction_score : float,
        w_rec               : float = W_RECONSTRUCTION,
        w_con               : float = W_CONTRADICTION,
    ) -> float:
        """Convenience static method."""
        return float(min(max(w_rec * norm_error + w_con * contradiction_score, 0.0), 1.0))


# ──────────────────────────────────────────────────────────────────────────
# Verdict helper
# ──────────────────────────────────────────────────────────────────────────

def get_verdict(final_score: float) -> tuple[str, str]:
    """
    Map [0,1] fake-score to a human verdict + emoji.

    Returns
    -------
    (verdict_text, emoji)
    """
    if final_score < VERDICT_GENUINE:
        return 'Likely Genuine',    EMOJI['genuine']
    elif final_score < VERDICT_SUSPICIOUS:
        return 'Suspicious',        EMOJI['suspicious']
    else:
        return 'Likely Fake',       EMOJI['fake']