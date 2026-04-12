#!/usr/bin/env python3
"""
ReviewGuard Full Pipeline Orchestration

Runs all training, evaluation, and experiment scripts in proper order.

Usage:
    python run_all.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd: str, description: str) -> bool:
    """
    Run shell command and return success status.
    
    Args:
        cmd: Command to execute
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} - Complete\n")
            return True
        else:
            print(f"❌ {description} - Failed\n")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}\n")
        return False


def main():
    """Execute full ReviewGuard pipeline."""
    
    print_section("🛡️  ReviewGuard - Full Pipeline")
    print("Starting complete training and evaluation pipeline...")
    print(f"Working directory: {os.getcwd()}\n")
    
    # Check necessary files
    print("📋 Checking project structure...")
    for path in ['models', 'training', 'utils', 'data', 'experiments', 'assets']:
        if Path(path).exists():
            print(f"  ✓ {path}/")
        else:
            print(f"  ✗ {path}/ - Missing!")
    print()
    
    # Pipeline stages
    stages = [
        {
            'name': 'Stage 1: Sentiment Classifier Training',
            'cmd': 'python training/train_sentiment.py',
            'time_est': '~1.5 hours (M2 GPU with MPS)'
        },
        {
            'name': 'Stage 2: Autoencoder Training',
            'cmd': 'python training/train_autoencoder.py',
            'time_est': '~30 min (M2 GPU with MPS)'
        },
        {
            'name': 'Optimizer Comparison Analysis',
            'cmd': 'python -m experiments.optimizer_comparison',
            'time_est': '~1 min'
        },
        {
            'name': 'Regularization Study Analysis',
            'cmd': 'python -m experiments.regularization_study',
            'time_est': '~1 min'
        }
    ]
    
    # Execute stages
    results = {}
    start_time = time.time()
    
    for i, stage in enumerate(stages, 1):
        print_section(f"Stage {i}/{len(stages)}: {stage['name']}")
        print(f"Estimated time: {stage['time_est']}\n")
        
        stage_start = time.time()
        success = run_command(stage['cmd'], stage['name'])
        stage_time = time.time() - stage_start
        
        results[stage['name']] = {
            'success': success,
            'time': stage_time
        }
        
        if not success and i < len(stages):  # Don't fail on experiments
            if i == 1:
                print("⚠️  Cannot proceed without Stage 1. Exiting...\n")
                sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    
    print_section("📊 Pipeline Complete - Summary")
    
    print(f"{'Stage':<40} {'Status':<12} {'Time':<10}")
    print("-" * 62)
    
    for stage_name, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        time_str = f"{result['time']/60:.1f}m"
        stage_short = stage_name[:37] + "..." if len(stage_name) > 40 else stage_name
        print(f"{stage_short:<40} {status:<12} {time_str:<10}")
    
    print(f"\nTotal pipeline time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    # Next steps
    print_section("🚀 Next Steps")
    
    if all(result['success'] for result in results.values() if 'Stage' in list(results.keys())[i]):
        print("""
✅ All training complete! Models are ready for deployment.

Next steps:

1. View model performance:
   streamlit run app.py
   
   Then navigate to "Model Performance" tab

2. Analyze reviews:
   - Single review: Use the main UI
   - Batch processing: Upload CSV with [review_text, star_rating] columns
   
3. Explore trained models:
   - Sentiment model: models/saved/sentiment_model_adam_best.pt
   - Autoencoder: models/saved/autoencoder_checkpoint.pt
   - Vocabulary: models/saved/vocabulary.pkl
   
4. Review training logs:
   - Sentiment: models/saved/training_log_sentiment.json
   - Autoencoder: models/saved/training_log_autoencoder.json
   
5. Experiment results:
   - Optimizer comparison: models/saved/optimizer_* (HTML + PNG)
   - Regularization study: models/saved/regularization_* (HTML + PNG)
        """)
    else:
        print("""
⚠️  Some stages failed. Check logs above.

To restart individual stages:

  python -m training.train_sentiment
  python -m training.train_autoencoder
  python -m experiments.optimizer_comparison
  python -m experiments.regularization_study
        """)
    
    print("\n" + "="*70)
    print("  ReviewGuard Pipeline - Complete")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
