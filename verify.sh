#!/bin/bash
# ReviewGuard - Project Verification & Setup Script

echo "═══════════════════════════════════════════════════════════"
echo "  ReviewGuard - Verification & Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Python
echo "🔍 Checking environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Python 3 found: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✓ Python found: $(python --version)"
else
    echo "✗ Python not found. Install Python 3.10+"
    exit 1
fi
echo ""

# Check project files
echo "📁 Checking project structure..."
files=(
    "app.py"
    "requirements.txt"
    "README.md"
    "QUICKSTART.md"
    "run_all.py"
    "models/sentiment_model.py"
    "models/autoencoder_model.py"
    "training/train_sentiment.py"
    "training/train_autoencoder.py"
    "utils/preprocessing.py"
    "utils/fusion.py"
    "utils/explainability.py"
    "experiments/optimizer_comparison.py"
    "experiments/regularization_study.py"
    "assets/styles.css"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ((missing++))
    fi
done

if [ $missing -eq 0 ]; then
    echo ""
    echo "✅ All project files present!"
else
    echo ""
    echo "⚠️  $missing files missing!"
fi
echo ""

# Suggest next steps
echo "═══════════════════════════════════════════════════════════"
echo "  Next Steps"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Run UI (no training needed, demo mode works):"
echo "   streamlit run app.py"
echo ""
echo "3. Train models (optional, ~3 hours on GPU):"
echo "   python run_all.py"
echo ""
echo "4. Read documentation:"
echo "   • QUICKSTART.md - Start here! (5 min read)"
echo "   • README.md - Full documentation"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
