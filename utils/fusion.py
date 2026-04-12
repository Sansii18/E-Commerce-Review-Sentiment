"""
Stage 3: Score Fusion for Final Fake Review Verdict

Combines signals from Stage 1 and Stage 2:
- Reconstruction error (primary signal)
- Contradiction score (rating vs sentiment mismatch)

Novel two-signal approach creates more robust fake detection
than using either signal alone.
"""

from typing import Tuple


class ScoreFuser:
    """
    Fuses reconstruction error and contradiction score into
    final fake review probability.
    """
    
    @staticmethod
    def compute_final_score(
        reconstruction_error: float,
        normalized_threshold: float,
        contradiction_score: float,
        weight_recon: float = 0.65,
        weight_contradiction: float = 0.35
    ) -> float:
        """
        Compute final fake review probability via weighted fusion.
        
        DESIGN RATIONALE:
        ─────────────────
        Reconstruction error (weight=0.65) is the primary signal:
        - Captures unusual word patterns, out-of-distribution language
        - Learned from 18K genuine reviews, detects deviation
        - Robust across review types and languages
        
        Contradiction score (weight=0.35) is secondary but highly
        discriminative for rating manipulation attacks:
        - 5★ with negative sentiment = obvious fraud signal
        - 1★ with positive sentiment = obvious fraud signal  
        - Weight=0.35 prevents false positives on ambiguous reviews
        
        Weights sum to 1.0 for interpretability (output ranges 0-1).
        
        Args:
            reconstruction_error: Anomaly score from autoencoder
            normalized_threshold: Threshold learned during calibration
            contradiction_score: Rating vs sentiment mismatch (0-1)
            weight_recon: Weight for reconstruction signal (0.65)
            weight_contradiction: Weight for contradiction signal (0.35)
            
        Returns:
            final_score: Probability of being fake (0.0 to 1.0)
        """
        # Normalize reconstruction error by threshold
        # If error equals threshold → normalization factor 1.0 → score contribution 0.5
        # If error > 2*threshold → clamped to 1.0 → max contribution
        normalized_error = min(
            reconstruction_error / (normalized_threshold * 2.0),
            1.0
        )
        
        # Fuse signals
        final_score = (
            weight_recon * normalized_error +
            weight_contradiction * contradiction_score
        )
        
        # Clamp to [0, 1]
        final_score = min(max(final_score, 0.0), 1.0)
        
        return final_score
    
    @staticmethod
    def get_verdict(final_score: float) -> Tuple[str, str]:
        """
        Convert numerical score to human-readable verdict.
        
        Args:
            final_score: Fake probability (0.0 to 1.0)
            
        Returns:
            Tuple of (verdict_text, color_code_hex)
        """
        if final_score < 0.35:
            return "Likely Genuine", "#10B981"  # Emerald green
        elif final_score < 0.60:
            return "Suspicious", "#F59E0B"  # Amber yellow
        else:
            return "Likely Fake", "#F43F5E"  # Rose red
    
    @staticmethod
    def get_verdict_icon(final_score: float) -> str:
        """
        Get emoji icon for verdict.
        
        Args:
            final_score: Fake probability
            
        Returns:
            Emoji character
        """
        if final_score < 0.35:
            return "✓"  # Check mark
        elif final_score < 0.60:
            return "⚠"  # Warning
        else:
            return "✗"  # X mark
    
    @staticmethod
    def explain_score(
        reconstruction_error: float,
        normalized_threshold: float,
        contradiction_score: float
    ) -> str:
        """
        Generate human-readable explanation of verdict.
        
        Args:
            reconstruction_error: Autoencoder error
            normalized_threshold: Calibrated threshold
            contradiction_score: Rating vs sentiment mismatch
            
        Returns:
            Explanation string
        """
        normalized_error = min(
            reconstruction_error / (normalized_threshold * 2.0),
            1.0
        )
        
        explanation_parts = []
        
        # Explain reconstruction error component
        if normalized_error > 0.7:
            explanation_parts.append(
                f"• Review uses unusual language patterns (error: {reconstruction_error:.3f} "
                f"vs threshold: {normalized_threshold:.3f})"
            )
        elif normalized_error > 0.4:
            explanation_parts.append(
                f"• Moderate pattern deviation detected"
            )
        else:
            explanation_parts.append(
                f"• Review language patterns appear normal"
            )
        
        # Explain contradiction component
        if contradiction_score > 0.6:
            explanation_parts.append(
                f"• Strong rating/sentiment contradiction detected "
                f"({contradiction_score:.1%} mismatch)"
            )
        elif contradiction_score > 0.3:
            explanation_parts.append(
                f"• Mild rating/sentiment mismatch detected"
            )
        else:
            explanation_parts.append(
                f"• Rating aligns with expressed sentiment"
            )
        
        return "\n".join(explanation_parts)
