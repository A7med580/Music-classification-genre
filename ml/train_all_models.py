"""
Train All Models — Unified Training Script.

Runs feature analysis, CNN training, GMM training, and model comparison
in sequence. Produces all required outputs in ml/outputs/.

Usage:
    python ml/train_all_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import ensure_output_dir


def main():
    """Execute the complete training pipeline."""
    print("\n" + "=" * 60)
    print("   MUSIC GENRE CLASSIFICATION — FULL TRAINING PIPELINE")
    print("=" * 60)

    ensure_output_dir()

    # Step 1: Feature Analysis
    print("\n\n>>> STEP 1: Feature Analysis <<<")
    print("-" * 40)
    from feature_analysis import main as run_feature_analysis
    run_feature_analysis()

    # Step 2: CNN Training
    print("\n\n>>> STEP 2: CNN Model Training <<<")
    print("-" * 40)
    from models.cnn_model import train_cnn
    cnn_results = train_cnn(epochs=50, batch_size=32)

    # Step 3: GMM Baseline Training
    print("\n\n>>> STEP 3: GMM Baseline Training <<<")
    print("-" * 40)
    from models.gmm_baseline import train_gmm
    gmm_results = train_gmm(n_components=16)

    # Step 3b: HMM Baseline Training
    print("\n\n>>> STEP 3b: HMM Baseline Training <<<")
    print("-" * 40)
    from models.hmm_baseline import train_hmm
    hmm_results = train_hmm(n_components=4)

    # Step 4: Model Comparison
    print("\n\n>>> STEP 4: Model Comparison <<<")
    print("-" * 40)
    from compare_models import main as run_comparison
    comparison = run_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("   ALL TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  CNN Accuracy:  {cnn_results['accuracy']*100:.1f}%")
    print(f"  GMM Accuracy:  {gmm_results['accuracy']*100:.1f}%")
    print(f"  HMM Accuracy:  {hmm_results['accuracy']*100:.1f}%")
    print(f"  Difference:    +{(cnn_results['accuracy']-max(gmm_results['accuracy'], hmm_results['accuracy']))*100:.1f} points (CNN vs best baseline)")
    print(f"\n  All outputs saved to: ml/outputs/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
