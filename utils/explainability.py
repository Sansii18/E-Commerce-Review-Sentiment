"""
Explainability Module: Attention-based Interpretation

Extracts interpretable signals from models:
1. Attention weights → sentiment-driving tokens
2. Reconstruction errors → tokens unusual for genuine reviews
3. HTML highlighting for interactive visualization

Helps users understand why ReviewGuard flags a review.
"""

from typing import List, Tuple
import numpy as np


class ExplainabilityEngine:
    """
    Generates human-interpretable explanations of model predictions.
    """
    
    @staticmethod
    def get_attention_highlights(
        tokens: List[str],
        attention_weights: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Extract tokens with highest attention weights.
        
        These are the words most responsible for sentiment prediction.
        High attention = crucial for determining if review is positive/negative.
        
        Args:
            tokens: List of token strings (from review)
            attention_weights: Numpy array of shape (seq_len,) with attention scores
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, weight) tuples sorted by weight descending
        """
        if len(tokens) == 0 or len(attention_weights) == 0:
            return []
        
        # Get indices of top-k weights
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        highlights = []
        for idx in top_indices:
            if idx < len(tokens):
                token = tokens[idx]
                weight = float(attention_weights[idx])
                # Skip padding token
                if token.strip() and token != '<PAD>':
                    highlights.append((token, weight))
        
        return highlights
    
    @staticmethod
    def get_suspicious_tokens(
        original_tokens: List[str],
        reconstructed_tokens: List[str],
        token_errors: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Find tokens with high reconstruction error.
        
        These tokens look "out of distribution" compared to genuine reviews.
        The autoencoder was trained on genuine reviews only, so high error
        on a token suggests it's unusual/suspicious.
        
        Args:
            original_tokens: List of original token strings
            reconstructed_tokens: List of reconstructed token strings
            token_errors: Numpy array of shape (seq_len,) with per-token errors
            
        Returns:
            List of (token, error) tuples where error > mean + 1*std
        """
        if len(token_errors) == 0:
            return []
        
        # Compute threshold: mean + 1 standard deviation
        mean_error = np.mean(token_errors)
        std_error = np.std(token_errors)
        threshold = mean_error + std_error
        
        # Find anomalous tokens
        suspicious = []
        for i, error in enumerate(token_errors):
            if error > threshold and i < len(original_tokens):
                token = original_tokens[i]
                if token.strip() and token != '<PAD>':
                    suspicious.append((token, float(error)))
        
        # Sort by error descending
        suspicious.sort(key=lambda x: x[1], reverse=True)
        
        return suspicious[:10]  # Return top 10
    
    @staticmethod
    def format_highlighted_html(
        tokens: List[str],
        weights: np.ndarray,
        mode: str = 'attention',
        max_weight: float = None
    ) -> str:
        """
        Generate HTML with inline highlights.
        
        Args:
            tokens: List of token strings
            weights: Numpy array of weights (same length as tokens)
            mode: 'attention' (yellow) or 'reconstruction' (red)
            max_weight: Maximum weight for normalization (auto if None)
            
        Returns:
            HTML string safe for st.markdown(unsafe_allow_html=True)
        """
        if len(tokens) == 0 or len(weights) == 0:
            return "<p>No tokens to highlight</p>"
        
        if max_weight is None:
            max_weight = np.max(weights)
        
        if max_weight == 0:
            max_weight = 1.0
        
        # Color schemes
        if mode == 'attention':
            # Yellow for sentiment signals
            color_low = "#FFFBEB"  # Very light yellow
            color_high = "#FCD34D"  # Medium yellow
        else:
            # Red for anomalies
            color_low = "#FEF2F2"  # Very light red
            color_high = "#FCA5A5"  # Medium red
        
        html_parts = []
        for token, weight in zip(tokens, weights):
            if token.strip() and token != '<PAD>':
                # Normalize weight to [0, 1]
                norm_weight = float(weight) / max_weight
                
                # Interpolate color
                r, g, b = ExplainabilityEngine._interpolate_color(
                    color_low, color_high, norm_weight
                )
                bg_color = f"rgb({r}, {g}, {b})"
                
                # Add space after token (unless it's punctuation)
                space = "" if token in ['!', '?', ',', '.', ';'] else " "
                
                html_parts.append(
                    f'<span style="background-color: {bg_color}; '
                    f'padding: 2px 4px; border-radius: 3px; '
                    f'font-weight: {500 + int(200 * norm_weight)}">'
                    f'{token}</span>{space}'
                )
        
        html = "<p>" + "".join(html_parts) + "</p>"
        return html
    
    @staticmethod
    def _interpolate_color(color1: str, color2: str, ratio: float) -> Tuple[int, int, int]:
        """
        Interpolate between two hex colors.
        
        Args:
            color1: Hex color like "#FFFBEB"
            color2: Hex color like "#FCD34D"
            ratio: Interpolation ratio (0 = color1, 1 = color2)
            
        Returns:
            Tuple of (R, G, B) values
        """
        # Parse hex colors
        c1 = int(color1[1:], 16)
        c2 = int(color2[1:], 16)
        
        r1, g1, b1 = (c1 >> 16) & 255, (c1 >> 8) & 255, c1 & 255
        r2, g2, b2 = (c2 >> 16) & 255, (c2 >> 8) & 255, c2 & 255
        
        # Interpolate
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        return r, g, b
    
    @staticmethod
    def format_pills_html(items: List[Tuple[str, float]]) -> str:
        """
        Format list of (label, score) as HTML pill badges.
        
        Args:
            items: List of (label, score) tuples
            
        Returns:
            HTML string with styled pills
        """
        if not items:
            return "<p><em>None detected</em></p>"
        
        pills = []
        for label, score in items[:5]:  # Show top 5
            # Determine pill color based on score
            if score > 0.7:
                bg_color = "#DBEAFE"  # Light blue
                text_color = "#1E40AF"  # Dark blue
            else:
                bg_color = "#E5E7EB"  # Light gray
                text_color = "#374151"  # Dark gray
            
            pills.append(
                f'<span style="display: inline-block; '
                f'background-color: {bg_color}; '
                f'color: {text_color}; '
                f'padding: 6px 12px; '
                f'border-radius: 20px; '
                f'margin-right: 8px; '
                f'font-size: 13px; '
                f'font-weight: 500;">'
                f'{label} ({score:.2%})</span>'
            )
        
        return "".join(pills)
    
    @staticmethod
    def create_gauge_svg(
        score: float,
        size: int = 200,
        show_percentage: bool = True
    ) -> str:
        """
        Create SVG gauge chart for fake probability score.
        
        Gauge fills from green (genuine) through yellow (suspicious) to red (fake).
        
        Args:
            score: Fake probability (0.0 to 1.0)
            size: SVG size in pixels
            show_percentage: If True, show score as percentage
            
        Returns:
            SVG string
        """
        # Gauge is an arc from 180° to 0° (bottom to top)
        # Score determines how full the arc is
        
        # Colors based on score
        if score < 0.35:
            color = "#10B981"  # Green
            label = "GENUINE"
        elif score < 0.60:
            color = "#F59E0B"  # Amber
            label = "SUSPICIOUS"
        else:
            color = "#F43F5E"  # Red
            label = "FAKE"
        
        # SVG parameters
        radius = 60
        circumference = 2 * 3.14159 * radius
        stroke_dashoffset = circumference * (1 - score)
        
        percentage_text = f"{int(score * 100)}" if show_percentage else ""
        
        svg = f'''
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <!-- Background arc -->
            <circle cx="{size//2}" cy="{size//2}" r="{radius}" 
                    fill="none" stroke="#E5E7EB" stroke-width="8"/>
            
            <!-- Gauge arc -->
            <circle cx="{size//2}" cy="{size//2}" r="{radius}" 
                    fill="none" stroke="{color}" stroke-width="8"
                    stroke-dasharray="{circumference}" 
                    stroke-dashoffset="{stroke_dashoffset}"
                    stroke-linecap="round"
                    transform="rotate(-180 {size//2} {size//2})"/>
            
            <!-- Center text -->
            <text x="{size//2}" y="{size//2}" 
                  text-anchor="middle" dy="0.3em"
                  font-size="48" font-weight="bold" fill="{color}">
                {percentage_text}
            </text>
            <text x="{size//2}" y="{size//2 + 35}" 
                  text-anchor="middle"
                  font-size="14" font-weight="600" fill="#6B7280">
                AUTHENTICITY
            </text>
        </svg>
        '''
        
        return svg.strip()
