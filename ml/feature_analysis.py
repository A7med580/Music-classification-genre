"""
Feature Analysis for Music Genre Classification.

Computes feature importance using mutual information, generates correlation
matrices, runs ablation studies, and produces comprehensive visualizations
to justify feature selection decisions.

Outputs:
    - Feature importance bar chart (PNG)
    - Correlation matrix heatmap (PNG)
    - Ablation study results (PNG + JSON)
    - Feature distribution plots per genre (PNG)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import load_raw_dataframe, ensure_output_dir, FEATURES_TO_EXCLUDE
from core.features import FEATURE_GROUPS, FEATURES_TO_USE, FEATURES_TO_EXCLUDE as EXCLUDE_GROUPS


def compute_mutual_information(df):
    """Compute mutual information between each feature and genre label.

    Args:
        df: DataFrame with features and 'label' column.

    Returns:
        pandas Series of MI scores indexed by feature name, sorted descending.
    """
    feature_cols = [c for c in df.columns if c not in ['filename', 'length', 'label']]
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df['label'].values)

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    return mi_series


def plot_feature_importance(mi_scores, output_dir):
    """Plot mutual information feature importance bar chart.

    Args:
        mi_scores: pandas Series of MI scores.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color bars by feature group
    colors = []
    for feat in mi_scores.index:
        if 'mfcc' in feat:
            colors.append('#2196F3')  # Blue for MFCCs
        elif 'centroid' in feat:
            colors.append('#4CAF50')  # Green for Spectral Centroid
        elif 'zero_crossing' in feat or 'zcr' in feat:
            colors.append('#FF9800')  # Orange for ZCR
        elif feat in FEATURES_TO_EXCLUDE or feat == 'tempo':
            colors.append('#F44336')  # Red for excluded
        else:
            colors.append('#9E9E9E')  # Gray for others

    bars = ax.barh(range(len(mi_scores)), mi_scores.values, color=colors)
    ax.set_yticks(range(len(mi_scores)))
    ax.set_yticklabels(mi_scores.index, fontsize=8)
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_title('Feature Importance — Mutual Information with Genre Label',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='MFCC (USED)'),
        Patch(facecolor='#4CAF50', label='Spectral Centroid (USED)'),
        Patch(facecolor='#FF9800', label='Zero Crossing Rate (USED)'),
        Patch(facecolor='#F44336', label='EXCLUDED features'),
        Patch(facecolor='#9E9E9E', label='Other features'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    print("  Feature importance plot saved.")


def plot_correlation_matrix(df, output_dir):
    """Plot feature correlation matrix heatmap.

    Args:
        df: DataFrame with feature columns.
        output_dir: Directory to save the plot.
    """
    feature_cols = [c for c in df.columns if c not in ['filename', 'length', 'label']]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.5},
                xticklabels=True, yticklabels=True)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print("  Correlation matrix saved.")


def plot_feature_distributions(df, output_dir):
    """Plot feature distributions across genres for key features.

    Shows violin plots for the top 6 most important features, colored by genre.

    Args:
        df: DataFrame with features and 'label' column.
        output_dir: Directory to save the plot.
    """
    key_features = [
        'mfcc1_mean', 'spectral_centroid_mean', 'zero_crossing_rate_mean',
        'mfcc2_mean', 'chroma_stft_mean', 'rolloff_mean'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        if feat in df.columns:
            sns.violinplot(data=df, x='label', y=feat, ax=axes[idx],
                          inner='quartile', cut=0, palette='Set3')
            axes[idx].set_title(feat, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle('Feature Distributions Across Genres', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150)
    plt.close()
    print("  Feature distribution plots saved.")


def run_ablation_study(df, output_dir):
    """Run ablation study: compare accuracy with/without feature groups.

    Trains a Random Forest classifier in each ablation condition and reports
    cross-validated accuracy. This validates our feature selection decisions.

    Args:
        df: DataFrame with features and 'label' column.
        output_dir: Directory to save results.

    Returns:
        Dictionary of ablation results.
    """
    print("\n  Running ablation study (this may take a few minutes)...")

    all_features = [c for c in df.columns if c not in ['filename', 'length', 'label']]
    X_all = df[all_features].values
    y = df['label'].values

    scaler = MinMaxScaler()

    results = {}

    # Baseline: all features
    X_scaled = scaler.fit_transform(X_all)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    results['All Features (57)'] = float(np.mean(scores))
    print(f"    All features: {np.mean(scores):.4f}")

    # Remove each excluded group
    for group_name in EXCLUDE_GROUPS:
        group = FEATURE_GROUPS[group_name]
        cols_to_remove = [c for c in group['columns'] if c in all_features]
        remaining = [c for c in all_features if c not in cols_to_remove]
        X_sub = df[remaining].values
        X_scaled = scaler.fit_transform(X_sub)
        scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
        results[f'Without {group_name} ({len(remaining)})'] = float(np.mean(scores))
        print(f"    Without {group_name}: {np.mean(scores):.4f}")

    # Remove all excluded features
    excluded_cols = FEATURES_TO_EXCLUDE
    selected = [c for c in all_features if c not in excluded_cols]
    X_sel = df[selected].values
    X_scaled = scaler.fit_transform(X_sel)
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    results[f'Selected Features Only ({len(selected)})'] = float(np.mean(scores))
    print(f"    Selected features only: {np.mean(scores):.4f}")

    # Plot ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    colors = ['#2196F3'] + ['#FF9800'] * len(EXCLUDE_GROUPS) + ['#4CAF50']
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel('Cross-Validated Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Feature Group Impact', fontsize=14, fontweight='bold')
    ax.set_xlim(min(values) - 0.02, max(values) + 0.02)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=150)
    plt.close()
    print("  Ablation study plot saved.")

    return results


def main():
    """Run the complete feature analysis pipeline."""
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    output_dir = ensure_output_dir()
    df = load_raw_dataframe()
    print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Genres: {sorted(df['label'].unique())}")

    # 1. Mutual Information
    print("\n[1/4] Computing mutual information scores...")
    mi_scores = compute_mutual_information(df)
    plot_feature_importance(mi_scores, output_dir)

    # Save MI scores
    mi_scores.to_csv(os.path.join(output_dir, 'mutual_information_scores.csv'))

    print("\n  Top 10 features by MI:")
    for feat, score in mi_scores.head(10).items():
        print(f"    {feat:30s} {score:.4f}")

    print("\n  Bottom 5 features by MI:")
    for feat, score in mi_scores.tail(5).items():
        print(f"    {feat:30s} {score:.4f}")

    # 2. Correlation Matrix
    print("\n[2/4] Generating correlation matrix...")
    plot_correlation_matrix(df, output_dir)

    # 3. Feature Distributions
    print("\n[3/4] Generating feature distribution plots...")
    plot_feature_distributions(df, output_dir)

    # 4. Ablation Study
    print("\n[4/4] Running ablation study...")
    ablation = run_ablation_study(df, output_dir)

    # Save summary
    summary = {
        'features_to_use': {
            name: {
                'columns': FEATURE_GROUPS[name]['columns'],
                'reason': FEATURE_GROUPS[name].get('reason_use', ''),
            }
            for name in FEATURES_TO_USE
        },
        'features_to_exclude': {
            name: {
                'columns': FEATURE_GROUPS[name]['columns'],
                'reason': FEATURE_GROUPS[name].get('reason_exclude', ''),
            }
            for name in EXCLUDE_GROUPS
        },
        'mutual_information_top10': {k: float(v) for k, v in mi_scores.head(10).items()},
        'ablation_study': ablation,
    }

    with open(os.path.join(output_dir, 'feature_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("FEATURE ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
