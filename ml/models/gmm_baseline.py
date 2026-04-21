"""
GMM (Gaussian Mixture Model) baseline for Music Genre Classification.

Implements a classical statistical approach: one GMM per genre.
During prediction, the genre whose GMM assigns the highest log-likelihood
to the input features is selected.

This serves as the baseline model to compare against the deep learning CNN.
Typical accuracy: 55-70% on GTZAN 3-second segments.
"""

import os
import sys
import json
import time
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import (
    load_dataset, prepare_data_split, ensure_output_dir
)


class GMMClassifier:
    """Multi-class classifier using one GMM per class.

    For each class, a separate Gaussian Mixture Model is trained on the
    class's training data. During prediction, the class whose GMM yields
    the highest log-likelihood for the input features is chosen.

    Attributes:
        n_components: Number of Gaussian components per GMM.
        covariance_type: Type of covariance ('full', 'diag', 'tied', 'spherical').
        gmms: Dictionary mapping class label to fitted GMM.
        classes: List of class labels.
    """

    def __init__(self, n_components=16, covariance_type='diag', random_state=42):
        """Initialize the GMM classifier.

        Args:
            n_components: Number of Gaussian components per GMM.
            covariance_type: Type of covariance parameters.
            random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmms = {}
        self.classes = []

    def fit(self, X, y):
        """Train one GMM per unique class label.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
        """
        self.classes = sorted(list(set(y)))

        for cls in self.classes:
            mask = (y == cls)
            X_cls = X[mask]

            # Adjust components if class has fewer samples
            n_comp = min(self.n_components, len(X_cls) // 2)
            n_comp = max(n_comp, 2)  # At least 2 components

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                max_iter=200,
                n_init=3,
                random_state=self.random_state
            )
            gmm.fit(X_cls)
            self.gmms[cls] = gmm

    def predict(self, X):
        """Predict class labels for input features.

        Computes log-likelihood under each class GMM and returns the
        class with the highest likelihood.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted class labels.
        """
        log_likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, cls in enumerate(self.classes):
            log_likelihoods[:, i] = self.gmms[cls].score_samples(X)

        predicted_indices = np.argmax(log_likelihoods, axis=1)
        return np.array([self.classes[idx] for idx in predicted_indices])

    def predict_proba(self, X):
        """Compute pseudo-probabilities from log-likelihoods.

        Applies softmax to log-likelihoods to get probability-like scores.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with pseudo-probabilities.
        """
        log_likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, cls in enumerate(self.classes):
            log_likelihoods[:, i] = self.gmms[cls].score_samples(X)

        # Softmax for probabilities
        exp_ll = np.exp(log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True))
        return exp_ll / np.sum(exp_ll, axis=1, keepdims=True)


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name='GMM'):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        output_dir: Directory to save the plot.
        model_name: Name for the plot title and filename.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Oranges',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()

    filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {output_dir}/{filename}")


def train_gmm(n_components=16):
    """Train the GMM classifier on GTZAN features and save results.

    Args:
        n_components: Number of Gaussian components per genre GMM.

    Returns:
        Dictionary with training results including accuracy and metrics.
    """
    print("=" * 60)
    print("GMM BASELINE MODEL TRAINING")
    print("=" * 60)

    output_dir = ensure_output_dir()

    # Load and prepare data
    print("\n[1/4] Loading dataset...")
    X, y, feature_names = load_dataset(exclude_features=True)
    data = prepare_data_split(X, y)
    print(f"  Features: {X.shape[1]} | Train: {data['X_train'].shape[0]} | Test: {data['X_test'].shape[0]}")

    # Train GMM
    print(f"\n[2/4] Training GMM ({n_components} components per genre)...")
    start_time = time.time()

    gmm_clf = GMMClassifier(n_components=n_components, covariance_type='diag')
    gmm_clf.fit(data['X_train'], data['y_train'])

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s")

    # Evaluate
    print("\n[3/4] Evaluating...")
    y_pred = gmm_clf.predict(data['X_test'])
    accuracy = accuracy_score(data['y_test'], y_pred)

    report = classification_report(
        data['y_test'], y_pred,
        target_names=data['class_names'], output_dict=True
    )
    report_str = classification_report(
        data['y_test'], y_pred,
        target_names=data['class_names']
    )

    print(f"\n  GMM Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n{report_str}")

    # Save model and plots
    print("\n[4/4] Saving model and plots...")

    with open(os.path.join(output_dir, 'gmm_model.pkl'), 'wb') as f:
        pickle.dump(gmm_clf, f)

    plot_confusion_matrix(
        data['y_test'], y_pred,
        data['class_names'], output_dir, 'GMM'
    )

    # Save metrics
    results = {
        'model': 'GMM (Gaussian Mixture Model)',
        'accuracy': float(accuracy),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_macro': float(report['macro avg']['f1-score']),
        'training_time_seconds': round(training_time, 1),
        'n_components': n_components,
        'per_genre': {
            name: {
                'precision': float(report[name]['precision']),
                'recall': float(report[name]['recall']),
                'f1': float(report[name]['f1-score']),
            }
            for name in data['class_names'] if name in report
        }
    }

    with open(os.path.join(output_dir, 'gmm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"GMM TRAINING COMPLETE — Accuracy: {accuracy*100:.1f}%")
    print(f"{'=' * 60}\n")

    return results


if __name__ == '__main__':
    train_gmm()
