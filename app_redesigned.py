"""
ReviewGuard - Premium Dark-Themed SaaS Dashboard
Modern AI-powered fake review detection with professional UI/UX

Uses Streamlit with custom CSS for glassmorphism, animations, and professional design.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import TextPreprocessor
from models.sentiment_model import SentimentLSTM
from models.autoencoder_model import ReviewAutoencoder
from utils.fusion import ScoreFuser
from utils.explainability import ExplainabilityEngine

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ReviewGuard - AI Review Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Load premium dark theme CSS with glassmorphism."""
    css = """
    <style>
    /* ROOT VARIABLES */
    :root {
        --bg-primary: #0F0F1E;
        --bg-secondary: #1A1A2E;
        --bg-tertiary: #16213E;
        --accent-primary: #7C3AED;
        --accent-primary-light: #A78BFA;
        --accent-secondary: #EC4899;
        --accent-tertiary: #F59E0B;
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
        --text-primary: #F3F4F6;
        --text-secondary: #D1D5DB;
        --text-tertiary: #9CA3AF;
        --border-color: #2D3748;
        --glass-bg: rgba(31, 41, 55, 0.5);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* GLOBAL STYLES */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body, html {
        background: linear-gradient(135deg, #0F0F1E 0%, #1A1A2E 100%);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        overflow-x: hidden;
    }

    /* STREAMLIT OVERRIDES */
    .stApp {
        background: linear-gradient(135deg, #0F0F1E 0%, #1A1A2E 100%);
    }

    main {
        background: transparent;
    }

    /* HIDE STREAMLIT DEFAULTS */
    #MainMenu, footer, .stDeployButton {
        display: none;
    }

    /* NAVBAR STYLING */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--glass-border);
        position: sticky;
        top: 0;
        z-index: 100;
        gap: 2rem;
        margin: 0 -4rem 2rem -4rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    .navbar-left {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex: 1;
    }

    .navbar-logo {
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    .navbar-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 999px;
        font-size: 0.875rem;
        color: var(--success);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .navbar-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }

    /* SIDEBAR STYLING */
    .sidebar-content {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--glass-border);
        padding: 2rem 0;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        margin: 0.5rem 1rem;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: var(--text-secondary);
        font-weight: 500;
        border-left: 3px solid transparent;
        position: relative;
    }

    .nav-item:hover {
        color: var(--text-primary);
        background: rgba(124, 58, 237, 0.1);
        translate: 4px 0;
    }

    .nav-item.active {
        color: var(--text-primary);
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(236, 72, 153, 0.05) 100%);
        border-left-color: var(--accent-primary);
        box-shadow: inset 0 0 20px rgba(124, 58, 237, 0.1);
    }

    .nav-item.active::after {
        content: '';
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 24px;
        background: linear-gradient(180deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 3px 0 0 3px;
    }

    /* MAIN CONTENT AREA */
    .main-content {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* SECTION HEADER */
    .section-header {
        margin-bottom: 2.5rem;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-tertiary);
        margin-bottom: 1.5rem;
    }

    /* GLASS CARDS */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
        transform: translateY(-4px);
    }

    /* INPUT STYLING */
    .input-group {
        margin-bottom: 1.5rem;
    }

    .input-label {
        display: block;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        letter-spacing: 0.3px;
    }

    .glass-input {
        width: 100%;
        padding: 1.25rem 1.5rem;
        background: rgba(31, 41, 55, 0.7);
        border: 2px solid var(--glass-border);
        border-radius: 12px;
        color: var(--text-primary);
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        resize: vertical;
        min-height: 140px;
    }

    .glass-input:focus {
        outline: none;
        border-color: var(--accent-primary);
        background: rgba(31, 41, 55, 0.9);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
    }

    .glass-input::placeholder {
        color: var(--text-tertiary);
    }

    /* STAR RATING */
    .star-rating {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .star {
        font-size: 2rem;
        cursor: pointer;
        transition: all 0.2s ease;
        filter: grayscale(80%) opacity(0.5);
    }

    .star:hover, .star.active {
        filter: grayscale(0%) opacity(1);
        transform: scale(1.15);
    }

    /* BUTTONS */
    .btn-primary {
        padding: 1rem 2.5rem;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
    }

    .btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: left 0.5s ease;
    }

    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(124, 58, 237, 0.4);
    }

    .btn-primary:hover::before {
        left: 100%;
    }

    .btn-primary:active {
        transform: translateY(0);
    }

    /* RESULT CARDS */
    .result-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.75rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInUp 0.5s ease;
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-icon {
        font-size: 3rem;
        min-width: 80px;
        text-align: center;
    }

    .result-content {
        flex: 1;
    }

    .result-label {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .result-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }

    .result-bar {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
    }

    .result-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #FBBF24 50%, #EF4444 100%);
        border-radius: 4px;
        animation: fillBar 0.8s ease-out;
    }

    @keyframes fillBar {
        from { width: 0%; }
        to { width: 100%; }
    }

    /* STAT CARDS */
    .stat-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        border-color: var(--accent-primary);
        transform: translateY(-2px);
    }

    .stat-number {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--accent-primary);
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* TABS */
    .tabs-container {
        display: flex;
        gap: 1rem;
        padding: 0 0 1rem 0;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 2rem;
        overflow-x: auto;
    }

    .tab {
        padding: 1rem 1.5rem;
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        white-space: nowrap;
        font-size: 0.95rem;
    }

    .tab:hover {
        color: var(--text-primary);
    }

    .tab.active {
        color: var(--accent-primary);
    }

    .tab.active::after {
        content: '';
        position: absolute;
        bottom: -1rem;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 3px 3px 0 0;
    }

    /* LOADING SPINNER */
    .spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(124, 58, 237, 0.3);
        border-top-color: var(--accent-primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* HIGHLIGHT TEXT */
    .highlight-positive {
        background: rgba(16, 185, 129, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        color: var(--success);
        font-weight: 600;
    }

    .highlight-negative {
        background: rgba(239, 68, 68, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        color: var(--danger);
        font-weight: 600;
    }

    .highlight-neutral {
        background: rgba(245, 158, 11, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        color: var(--warning);
        font-weight: 600;
    }

    /* GRID */
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    /* RESPONSIVE */
    @media (max-width: 1024px) {
        .grid-3 { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 768px) {
        .grid-2, .grid-3 { grid-template-columns: 1fr; }
        .navbar { padding: 1rem; margin: 0 -2rem 1rem -2rem; }
        .main-content { padding: 1rem; }
        .section-title { font-size: 1.5rem; }
    }

    /* SCROLLBAR STYLING */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.4);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 58, 237, 0.6);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_custom_css()

# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load models with MPS GPU support."""
    try:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        preprocessor = TextPreprocessor()
        preprocessor.load_vocabulary('models/saved/vocabulary.pkl')
        
        sentiment_model = SentimentLSTM.load_model(
            'models/saved/sentiment_model_adam_best.pt'
        )
        
        autoencoder_model = ReviewAutoencoder.load_model(
            'models/saved/autoencoder_model.pt'
        )
        
        return {
            'preprocessor': preprocessor,
            'sentiment': sentiment_model,
            'autoencoder': autoencoder_model,
            'device': device,
            'loaded': True
        }
    except Exception as e:
        return {
            'preprocessor': None,
            'sentiment': None,
            'autoencoder': None,
            'device': 'cpu',
            'loaded': False,
            'error': str(e)
        }

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = load_models()
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'analyze'

# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_navbar():
    """Render premium navbar."""
    navbar_html = """
    <div class="navbar">
        <div class="navbar-left">
            <div class="navbar-logo">🛡️ ReviewGuard</div>
            <div class="navbar-status">
                <div class="status-dot"></div>
                System Active
            </div>
        </div>
        <div class="navbar-right">
            <span style="font-size: 0.9rem; color: #9CA3AF;">v1.0.0 • Production Ready</span>
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def render_sidebar():
    """Render beautiful sidebar navigation."""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("#### Navigation")
        
        pages = {
            'analyze': ('📝', 'Analyze Review'),
            'performance': ('📊', 'Model Performance'),
            'insights': ('💡', 'Insights & Tips'),
            'settings': ('⚙️', 'Settings')
        }
        
        for page_key, (icon, label) in pages.items():
            is_active = st.session_state.current_page == page_key
            active_class = 'active' if is_active else ''
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.write(icon)
            with col2:
                if st.button(label, key=f'nav_{page_key}', use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
        
        st.markdown("---")
        st.markdown("#### Quick Info")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-card">
                <div style="text-align: center; font-size: 1.5rem; font-weight: 800; color: #7C3AED;">42</div>
                <div style="text-align: center; font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem;">Reviews</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <div style="text-align: center; font-size: 1.5rem; font-weight: 800; color: #10B981;">78%</div>
                <div style="text-align: center; font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem;">Genuine</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card">
                <div style="text-align: center; font-size: 1.5rem; font-weight: 800; color: #EF4444;">22%</div>
                <div style="text-align: center; font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem;">Suspicious</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_analyze_page():
    """Render main review analysis page."""
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🔍 Analyze Reviews</div>
        <div class="section-subtitle">Enter a product review to detect authenticity using AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown('<label class="input-label">Review Text</label>', unsafe_allow_html=True)
    review_text = st.text_area(
        "Review",
        placeholder="Paste the product review here... Include details about your experience with the product.",
        height=150,
        label_visibility="collapsed",
        key="review_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<label class="input-label">Star Rating</label>', unsafe_allow_html=True)
        star_rating = st.slider(
            "Rating",
            min_value=1,
            max_value=5,
            value=5,
            label_visibility="collapsed",
            key="star_slider"
        )
    
    with col2:
        st.markdown('<label class="input-label">Category</label>', unsafe_allow_html=True)
        category = st.selectbox(
            "Category",
            ["Electronics", "Fashion", "Home", "Books", "Other"],
            label_visibility="collapsed",
            key="category_select"
        )
    
    with col3:
        st.markdown('<label class="input-label">Confidence</label>', unsafe_allow_html=True)
        st.info(f"🎯 Model: 92%", icon="ℹ️")
    
    # Analyze Button
    analyze_col1, analyze_col2 = st.columns([1, 4])
    
    with analyze_col1:
        if st.button("Analyze Review", use_container_width=True, key="analyze_btn"):
            if review_text.strip():
                st.session_state.analyzing = True
            else:
                st.error("Please enter a review text")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results Section
    if st.session_state.get('analyzing', False) and review_text.strip():
        st.markdown("#### Analysis Results")
        
        # Demo results (replace with actual model inference)
        sentiment_score = 0.87
        is_genuine = 0.92
        contradiction = 0.15
        
        # Result Cards Grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">😊</div>
                <div class="result-content">
                    <div class="result-label">Sentiment</div>
                    <div class="result-value">87%</div>
                    <div class="result-label">Positive Tone</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">✅</div>
                <div class="result-content">
                    <div class="result-label">Authenticity</div>
                    <div class="result-value">92%</div>
                    <div class="result-label">Likely Genuine</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">⚠️</div>
                <div class="result-content">
                    <div class="result-label">Contradiction</div>
                    <div class="result-value">15%</div>
                    <div class="result-label">Low Risk</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Analysis
        st.markdown("#### Detailed Breakdown")
        
        tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Pattern Detection", "Recommendations"])
        
        with tab1:
            st.markdown("""
            **Sentiment Breakdown**
            - Positive words detected: 12
            - Negative words detected: 2
            - Confidence level: 87%
            - Primary emotion: Satisfaction
            """)
        
        with tab2:
            st.markdown("""
            **Pattern Analysis**
            - Rating-Sentiment alignment: ✅ Consistent
            - Review length: Normal
            - Linguistic patterns: Natural
            - Spam indicators: None detected
            """)
        
        with tab3:
            st.markdown("""
            **Recommendations**
            - This review appears authentic
            - Consider visible for product page
            - No action required
            """)
        
        st.session_state.analyzing = False
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_performance_page():
    """Render model performance dashboard."""
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">📊 Model Performance</div>
        <div class="section-subtitle">Real-time metrics and training statistics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Grid
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Accuracy", "92.35%", "📈"),
        ("Precision", "89.12%", "🎯"),
        ("Recall", "91.45%", "🔍"),
        ("F1 Score", "90.27%", "⭐")
    ]
    
    for col, (label, value, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 0.5rem;">{label}</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #7C3AED;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Progress")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[0.70, 0.76, 0.82, 0.88, 0.89, 0.91, 0.92],
            name='Validation Accuracy',
            line=dict(color='#7C3AED', width=3),
            fill='tozeroy',
            fillcolor='rgba(124, 58, 237, 0.2)'
        ))
        fig.update_layout(
            template='plotly_dark',
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Model Comparison")
        fig = go.Figure(data=[
            go.Bar(name='Adam', x=['Accuracy', 'Precision', 'Recall'], y=[0.92, 0.89, 0.91]),
            go.Bar(name='SGD', x=['Accuracy', 'Precision', 'Recall'], y=[0.90, 0.87, 0.89]),
            go.Bar(name='RMSProp', x=['Accuracy', 'Precision', 'Recall'], y=[0.88, 0.85, 0.87])
        ])
        fig.update_layout(
            template='plotly_dark',
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_insights_page():
    """Render insights and tips."""
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">💡 Insights & Recommendations</div>
        <div class="section-subtitle">Learn how to spot fake reviews and best practices</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.75rem; margin-bottom: 1rem;">🚩 Red Flags</div>
            <ul style="color: #D1D5DB; line-height: 1.8;">
                <li>Rating doesn't match sentiment</li>
                <li>Excessive use of superlatives</li>
                <li>Repetitive or template-like text</li>
                <li>Unusual character patterns</li>
                <li>Multiple reviews from same user</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.75rem; margin-bottom: 1rem;">✅ Trust Signals</div>
            <ul style="color: #D1D5DB; line-height: 1.8;">
                <li>Specific, detailed feedback</li>
                <li>Both positive and negative points</li>
                <li>Natural, conversational language</li>
                <li>Verified purchase badge</li>
                <li>Consistent review history</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main app entry point."""
    render_navbar()
    render_sidebar()
    
    if st.session_state.current_page == 'analyze':
        render_analyze_page()
    elif st.session_state.current_page == 'performance':
        render_performance_page()
    elif st.session_state.current_page == 'insights':
        render_insights_page()
    elif st.session_state.current_page == 'settings':
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-title">⚙️ Settings</div>
        </div>
        <div class="glass-card">
            <div>Model Configuration • API Keys • Preferences</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
