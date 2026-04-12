"""
ReviewGuard - AI-Powered Review Authenticity Analyzer

Streamlit web application for:
1. Binary sentiment classification (Stage 1)
2. Fake review detection via LSTM autoencoder (Stage 2)
3. Score fusion combining both signals
4. Interactive visualization with attention highlights

Features:
- Single review analysis with detailed explanations
- Batch processing of CSV files
- Model performance dashboard
- Demo mode for UI demonstration without trained models
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from typing import Dict, Tuple, List
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from utils.preprocessing import TextPreprocessor
from utils.fusion import ScoreFuser
from utils.explainability import ExplainabilityEngine
from models.sentiment_model import SentimentLSTM
from models.autoencoder_model import ReviewAutoencoder

# ═════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ReviewGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═════════════════════════════════════════════════════════════

DEMO_MODE = not (
    Path("models/saved/sentiment_model_adam_best.pt").exists() and
    Path("models/saved/autoencoder_checkpoint.pt").exists()
)

CONFIG = {
    'max_vocab': 20000,
    'max_len': 200,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'device': (
        'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
}

# Demo reviews with pre-computed scores
DEMO_REVIEWS = [
    {
        'review': "This product is absolutely amazing! Exceeded all my expectations. Perfect quality and arrived in 2 days. Highly recommend!",
        'rating': 5,
        'sentiment': 0.94,
        'sentiment_label': 1,
        'reconstruction_error': 0.156,
        'contradiction': 0.0,
        'final_score': 0.18
    },
    {
        'review': "BEST PRODUCT EVER!!! MUST BUY NOW!!! Everyone should purchase immediately!!!",
        'rating': 5,
        'sentiment': 0.76,
        'sentiment_label': 1,
        'reconstruction_error': 0.834,
        'contradiction': 0.0,
        'final_score': 0.72
    },
    {
        'review': "Terrible product, complete waste of money. Broke after one day.",
        'rating': 1,
        'sentiment': 0.12,
        'sentiment_label': 0,
        'reconstruction_error': 0.234,
        'contradiction': 0.0,
        'final_score': 0.22
    },
    {
        'review': "Amazing product! Works perfectly! Best purchase ever made! Outstanding quality!",
        'rating': 1,
        'sentiment': 0.88,
        'sentiment_label': 1,
        'reconstruction_error': 0.412,
        'contradiction': 0.88,
        'final_score': 0.68
    },
    {
        'review': "Not great. Quality is poor and shipping was slow.",
        'rating': 2,
        'sentiment': 0.23,
        'sentiment_label': 0,
        'reconstruction_error': 0.312,
        'contradiction': 0.0,
        'final_score': 0.31
    }
]

# ═════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═════════════════════════════════════════════════════════════

if 'page' not in st.session_state:
    st.session_state.page = "Analyze Review"

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.sentiment_model = None
    st.session_state.autoencoder_model = None
    st.session_state.preprocessor = None
    st.session_state.threshold = None

if 'demo_review_index' not in st.session_state:
    st.session_state.demo_review_index = 0

# ═════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load trained models and preprocessor (cached)."""
    preprocessor = TextPreprocessor()
    
    if not DEMO_MODE:
        try:
            # Load sentiment model
            sentiment_model = SentimentLSTM.load_model(
                'models/saved/sentiment_model_adam_best.pt',
                device=CONFIG['device']
            )
            
            # Load autoencoder and threshold
            checkpoint = torch.load(
                'models/saved/autoencoder_checkpoint.pt',
                map_location=CONFIG['device']
            )
            autoencoder_model = ReviewAutoencoder.load_model(
                'models/saved/autoencoder_checkpoint.pt',
                device=CONFIG['device']
            )
            threshold = checkpoint['threshold']
            
            # Load vocabulary
            preprocessor.load_vocabulary('models/saved/vocabulary.pkl')
            
            return sentiment_model, autoencoder_model, preprocessor, threshold
        except Exception as e:
            st.warning(f"Could not load models: {e}. Running in demo mode.")
            return None, None, preprocessor, None
    
    return None, None, preprocessor, None

# ═════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═════════════════════════════════════════════════════════════

def analyze_review(
    review_text: str,
    star_rating: int,
    sentiment_model,
    autoencoder_model,
    preprocessor,
    threshold
) -> Dict:
    """
    Analyze a single review end-to-end.
    
    Returns:
        Dictionary with all analysis results
    """
    if DEMO_MODE:
        # Return demo results
        demo = DEMO_REVIEWS[st.session_state.demo_review_index]
        return {
            'review': review_text,
            'rating': star_rating,
            'sentiment_prob': demo['sentiment'],
            'sentiment_label': demo['sentiment_label'],
            'reconstruction_error': demo['reconstruction_error'],
            'contradiction_score': demo['contradiction'],
            'final_score': demo['final_score'],
            'attention_weights': np.random.rand(len(review_text.split())),
            'verdict': 'Demo Result',
            'is_fake': demo['final_score'] > 0.6
        }
    
    device = CONFIG['device']
    
    # Stage 1: Sentiment Analysis
    clean_review = preprocessor.clean_text(review_text)
    encoded = preprocessor.encode_text(clean_review, CONFIG['max_len'])
    
    with torch.no_grad():
        tokens_tensor = torch.LongTensor([encoded]).to(device)
        sentiment_prob, attention_weights = sentiment_model(tokens_tensor, return_attention=True)
        sentiment_prob = sentiment_prob.cpu().item()
        sentiment_label = 1 if sentiment_prob > 0.5 else 0
        attention_weights = attention_weights.squeeze().cpu().numpy()
    
    # Compute contradiction score
    contradiction_score = preprocessor.compute_contradiction_score(
        star_rating, sentiment_prob, sentiment_label
    )
    
    # Stage 2: Autoencoder Anomaly Detection
    with torch.no_grad():
        reconstructed = autoencoder_model(tokens_tensor)
        mask = (tokens_tensor != 0).float()
        reconstruction_error = autoencoder_model.compute_reconstruction_error(
            tokens_tensor, reconstructed, mask
        ).cpu().item()
    
    # Stage 3: Score Fusion
    normalized_threshold = threshold if threshold else 0.5
    final_score = ScoreFuser.compute_final_score(
        reconstruction_error,
        normalized_threshold,
        contradiction_score
    )
    
    verdict, color = ScoreFuser.get_verdict(final_score)
    
    return {
        'review': review_text,
        'rating': star_rating,
        'sentiment_prob': sentiment_prob,
        'sentiment_label': sentiment_label,
        'reconstruction_error': reconstruction_error,
        'contradiction_score': contradiction_score,
        'final_score': final_score,
        'attention_weights': attention_weights,
        'verdict': verdict,
        'color': color,
        'is_fake': final_score > 0.60
    }

# ═════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═════════════════════════════════════════════════════════════

def render_header():
    """Render custom header with logo and status."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0F0F13 0%, #1A1A24 100%);
        border-bottom: 1px solid #2A2A3A;
        padding: 2rem 3rem;
        margin: -2rem -3rem 2rem -3rem;
        border-radius: 0 0 16px 16px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 32px; display: flex; align-items: center; gap: 12px;">
                    🛡️ ReviewGuard
                </h1>
                <p style="margin: 8px 0 0 0; color: #9CA3AF; font-size: 14px;">
                    AI-powered review authenticity analysis powered by deep learning
                </p>
            </div>
            <div style="display: flex; gap: 12px;">
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.3);
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 12px;
                    color: #10B981;
                    display: flex; align-items: center; gap: 6px;
                ">
                    <span style="width: 6px; height: 6px; background: #10B981; border-radius: 50%; display: inline-block;"></span>
                    Sentiment: Ready
                </div>
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.3);
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 12px;
                    color: #10B981;
                    display: flex; align-items: center; gap: 6px;
                ">
                    <span style="width: 6px; height: 6px; background: #10B981; border-radius: 50%; display: inline-block;"></span>
                    Detector: Ready
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sentiment_card(sentiment_prob: float, truncate: bool = False):
    """Render sentiment analysis card."""
    sentiment_label = "POSITIVE" if sentiment_prob > 0.5 else "NEGATIVE"
    sentiment_emoji = "😊" if sentiment_prob > 0.5 else "😞"
    color = "#10B981" if sentiment_prob > 0.5 else "#F43F5E"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1A24 0%, #2A2A3A 100%);
        border: 1px solid #2A2A3A;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    ">
        <div style="font-size: 40px; margin-bottom: 0.5rem;">{sentiment_emoji}</div>
        <div style="
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9CA3AF;
            margin-bottom: 0.5rem;
        ">Sentiment</div>
        <div style="
            font-size: 24px;
            font-weight: 700;
            color: {color};
            margin-bottom: 1rem;
        ">{sentiment_label}</div>
        <div style="
            background: linear-gradient(90deg, #2A2A3A 0%, {color}40 {sentiment_prob*100}%, #2A2A3A {sentiment_prob*100}%, #2A2A3A 100%);
            border-radius: 4px;
            height: 8px;
            margin-bottom: 0.5rem;
        "></div>
        <div style="font-size: 13px; color: #D1D5DB;">
            {sentiment_prob*100:.1f}% confident
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_authenticity_card(final_score: float):
    """Render authenticity/fake probability card."""
    verdict, color = ScoreFuser.get_verdict(final_score)
    gauge_svg = ExplainabilityEngine.create_gauge_svg(final_score, size=200)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1A24 0%, #2A2A3A 100%);
        border: 1px solid #2A2A3A;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    ">
        <div style="margin-bottom: 1rem;">
            {gauge_svg}
        </div>
        <div style="
            display: inline-block;
            background: {color}20;
            color: {color};
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        ">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

def render_contradiction_card(rating: int, sentiment_prob: float, contradiction_score: float):
    """Render rating vs sentiment analysis card."""
    sentiment_label = "Positive" if sentiment_prob > 0.5 else "Negative"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1A24 0%, #2A2A3A 100%);
        border: 1px solid #2A2A3A;
        border-radius: 12px;
        padding: 1.5rem;
    ">
        <div style="
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9CA3AF;
            margin-bottom: 1rem;
        ">Rating vs Sentiment</div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 20px; color: #FCD34D;">★★★★★</div>
                <div style="font-size: 12px; color: #9CA3AF; margin-top: 0.25rem;">{rating} Rating</div>
            </div>
            <div style="color: #6B7280;">vs</div>
            <div style="text-align: center;">
                <div style="font-size: 14px; color: #6366F1; font-weight: 600;">{sentiment_label}</div>
                <div style="font-size: 12px; color: #9CA3AF; margin-top: 0.25rem;">Sentiment</div>
            </div>
        </div>
        
        {f'<div style="background: rgba(244, 63, 94, 0.1); border-left: 3px solid #F43F5E; padding: 12px; border-radius: 4px; margin-bottom: 1rem;"><div style="color: #F43F5E; font-weight: 600; font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">⚠️ Mismatch</div><div style="color: #D1D5DB; font-size: 13px;">{rating}★ with {sentiment_label.lower()} sentiment</div></div>' if contradiction_score > 0.3 else '<div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; padding: 12px; border-radius: 4px;"><div style="color: #10B981; font-weight: 600; font-size: 12px; text-transform: uppercase;">✓ Aligned</div></div>'}
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE: SINGLE REVIEW ANALYSIS
# ═════════════════════════════════════════════════════════════

def page_analyze_review():
    """Main analysis page."""
    render_header()
    
    if DEMO_MODE:
        st.info(
            "🎬 **Demo Mode Active** - Using pre-computed examples. "
            "Train models with `python -m training.train_sentiment` and "
            "`python -m training.train_autoencoder` to use live inference.",
            icon="ℹ️"
        )
    
    # Input section
    st.markdown("""
    <div style="
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    ">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <label style="
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9CA3AF;
        display: block;
        margin-bottom: 0.75rem;
    ">Paste a review to analyze</label>
    """, unsafe_allow_html=True)
    
    review_text = st.text_area(
        "Review text",
        placeholder="e.g., This product exceeded all my expectations! Arrived quickly and works perfectly...",
        height=140,
        label_visibility="collapsed"
    )
    
    # Rating and category selectors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        star_rating = st.select_slider(
            "Star Rating",
            options=[1, 2, 3, 4, 5],
            value=4
        )
    
    with col2:
        category = st.selectbox(
            "Category",
            ["Electronics", "Clothing", "Books", "Home", "Sports"],
            label_visibility="collapsed"
        )
    
    with col3:
        review_source = st.radio(
            "Source",
            ["Single", "Batch"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Review", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results section (only show after analysis)
    if analyze_button and review_text.strip():
        # Load models if not already loaded
        if not st.session_state.models_loaded:
            with st.spinner("Loading models..."):
                (st.session_state.sentiment_model,
                 st.session_state.autoencoder_model,
                 st.session_state.preprocessor,
                 st.session_state.threshold) = load_models()
                st.session_state.models_loaded = True
        
        # Show processing animation
        with st.container():
            processing_placeholder = st.empty()
            
            steps = ["Analyzing review...", "Checking patterns...", "Computing score..."]
            for step in steps:
                processing_placeholder.info(step, icon="⏳")
                time.sleep(0.5)
            
            processing_placeholder.empty()
        
        # Run analysis
        results = analyze_review(
            review_text,
            star_rating,
            st.session_state.sentiment_model,
            st.session_state.autoencoder_model,
            st.session_state.preprocessor,
            st.session_state.threshold
        )
        
        # Display results in three columns
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Analysis Results", divider="violet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_sentiment_card(results['sentiment_prob'])
        
        with col2:
            render_authenticity_card(results['final_score'])
        
        with col3:
            render_contradiction_card(star_rating, results['sentiment_prob'], results['contradiction_score'])
        
        # Explanation
        st.markdown("<br>", unsafe_allow_html=True)
        explanation = ScoreFuser.explain_score(
            results['reconstruction_error'],
            st.session_state.threshold if st.session_state.threshold else 0.5,
            results['contradiction_score']
        )
        
        st.markdown(f"""
        <div style="
            background: rgba(99, 102, 241, 0.05);
            border-left: 3px solid #6366F1;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        ">
            <strong style="color: #6366F1;">Why this verdict?</strong><br>
            <pre style="
                white-space: pre-wrap;
                font-family: 'Inter', sans-serif;
                font-size: 13px;
                color: #D1D5DB;
                margin: 0.75rem 0 0 0;
            ">{explanation}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Highlighted review text (tabs)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Review Text Analysis", divider="violet")
        
        tab1, tab2 = st.tabs(["Sentiment Signals", "Authenticity Signals"])
        
        with tab1:
            tokens = review_text.split()
            if len(results['attention_weights']) > 0:
                # Pad or trim attention weights to match tokens
                attention = results['attention_weights'][:len(tokens)]
                attention = np.pad(attention, (0, max(0, len(tokens) - len(attention))), mode='constant')
                
                highlighted_html = ExplainabilityEngine.format_highlighted_html(
                    tokens, attention, mode='attention'
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                st.text(review_text)
        
        with tab2:
            st.text(review_text)
            st.info("In production mode, this shows tokens with unusual patterns detected by the autoencoder.")


# ═════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════

def page_model_performance():
    """Model performance dashboard."""
    render_header()
    
    st.subheader("Model Performance", divider="violet")
    
    # Load training logs
    try:
        with open('models/saved/training_log_sentiment.json') as f:
            sentiment_log = json.load(f)
    except:
        st.warning("Sentiment training logs not found. Run training first.")
        sentiment_log = None
    
    try:
        with open('models/saved/training_log_autoencoder.json') as f:
            autoencoder_log = json.load(f)
    except:
        st.warning("Autoencoder training logs not found. Run training first.")
        autoencoder_log = None
    
    # Stage 1 results
    st.markdown("### Stage 1: Sentiment Classifier")
    
    if sentiment_log and 'best_model' in sentiment_log:
        metrics = sentiment_log['best_model']['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Precision", f"{metrics['precision']:.1%}")
        col3.metric("Recall", f"{metrics['recall']:.1%}")
        col4.metric("F1 Score", f"{metrics['f1']:.1%}")
    
    # Optimizer comparison
    if sentiment_log:
        st.markdown("**Optimizer Comparison:**")
        
        optimizers_data = []
        for opt in ['adam', 'sgd', 'rmsprop']:
            if opt in sentiment_log:
                log = sentiment_log[opt]
                optimizers_data.append({
                    'Optimizer': opt.upper(),
                    'Best Val Acc': log.get('best_val_accuracy', 0)
                })
        
        if optimizers_data:
            df_opt = pd.DataFrame(optimizers_data)
            fig = px.bar(
                df_opt, x='Optimizer', y='Best Val Acc',
                title='Validation Accuracy by Optimizer',
                template='plotly_dark',
                color_discrete_sequence=['#6366F1']
            )
            fig.update_layout(
                plot_bgcolor='rgba(26, 26, 36, 1)',
                paper_bgcolor='rgba(15, 15, 19, 1)',
                font_color='#E5E7EB'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Stage 2 results
    st.markdown("### Stage 2: Fake Review Detector")
    
    if autoencoder_log and 'metrics' in autoencoder_log:
        metrics = autoencoder_log['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        col2.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
        col3.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        col4.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        
        if 'threshold' in autoencoder_log:
            st.info(f"Threshold: {autoencoder_log['threshold']:.4f}")


# ═════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════

def page_about():
    """About page."""
    render_header()
    
    st.markdown("""
    ## About ReviewGuard
    
    ReviewGuard is a two-stage deep learning system for detecting fake product reviews:
    
    ### Stage 1: Sentiment Classifier
    - **Architecture**: 4-layer LSTM with single-head attention mechanism
    - **Training Data**: 200,000 Amazon reviews
    - **Task**: Binary classification (Positive/Negative)
    - **Output**: Confidence score + attention weights for interpretability
    
    ### Stage 2: Autoencoder Anomaly Detector
    - **Architecture**: LSTM Encoder-Decoder with latent bottleneck
    - **Training Data**: 18,000 genuine reviews (unsupervised)
    - **Task**: Detect out-of-distribution (fake) reviews
    - **Output**: Reconstruction error + threshold-based classification
    
    ### Novel Contribution
    ReviewGuard introduces a **contradiction score** combining stage 1 with stage 2:
    - High star rating + negative sentiment = suspicious (likely fake)
    - Low star rating + positive sentiment = suspicious (likely fake)
    - This signal, fused with reconstruction error, creates a more robust detector
    
    ### Score Fusion
    Final fake probability =  0.65 × normalized_reconstruction_error + 0.35 × contradiction_score
    
    ---
    
    **Model Status**: {"🟢 Live Models Ready" if not DEMO_MODE else "🎬 Demo Mode (models not trained)"}
    """)


# ═════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Analyze Review", "Model Performance", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Model Status")
    
    if DEMO_MODE:
        st.warning("🎬 Running in demo mode", icon="ℹ️")
    else:
        st.success("🟢 All models loaded", icon="✅")
    
    st.markdown("### Model Info")
    
    with st.expander("Stage 1: Sentiment"):
        st.markdown("""
        **LSTM Sentiment Classifier**
        - 4 stacked LSTM layers
        - Hidden dim: 256
        - Attention mechanism
        - Trained on 200K Amazon reviews
        """)
    
    with st.expander("Stage 2: Fake Detector"):
        st.markdown("""
        **LSTM Autoencoder**
        - Encoder-Decoder architecture
        - Hidden dim: 128
        - Trained on 18K genuine reviews
        - Unsupervised anomaly detection
        """)

# ═════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═════════════════════════════════════════════════════════════

if page == "Analyze Review":
    page_analyze_review()
elif page == "Model Performance":
    page_model_performance()
elif page == "About":
    page_about()
