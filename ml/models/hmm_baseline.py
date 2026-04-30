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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
try:
    from hmmlearn import hmm
except ImportError:
    print("Please install hmmlearn: pip install hmmlearn")

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import load_dataset, prepare_data_split, ensure_output_dir


class HMMClassifier:
    """Multi-class classifier using one HMM per class.
    
    Treats the aggregated feature vectors as single-observation sequences.
    While HMMs are best for temporal sequences, this fulfills the baseline
    requirement by modeling the statistical distribution over hidden states.
    """
    def __init__(self, n_components=4, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.hmms = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(list(set(y)))
        
        for cls in self.classes:
            mask = (y == cls)
            X_cls = X[mask]
            
            n_comp = min(self.n_components, len(X_cls) // 2)
            n_comp = max(n_comp, 1)

            model = hmm.GaussianHMM(
                n_components=n_comp, 
                covariance_type="diag", 
                n_iter=100,
                random_state=self.random_state
            )
            
            lengths = [1] * len(X_cls)
            model.fit(X_cls, lengths)
            self.hmms[cls] = model

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            sample = X[i:i+1]
            best_score = -np.inf
            best_cls = self.classes[0]
            
            for cls in self.classes:
                try:
                    score = self.hmms[cls].score(sample)
                    if score > best_score:
                        best_score = score
                        best_cls = cls
                except:
                    pass
            predictions.append(best_cls)
            
        return np.array(predictions)

def train_hmm(n_components=4):
    print("=" * 60)
    print("HMM BASELINE MODEL TRAINING")
    print("=" * 60)

    output_dir = ensure_output_dir()

    print("\n[1/4] Loading dataset...")
    X, y, feature_names = load_dataset(exclude_features=True)
    data = prepare_data_split(X, y)

    print(f"\n[2/4] Training HMM ({n_components} components per genre)...")
    start_time = time.time()

    hmm_clf = HMMClassifier(n_components=n_components)
    hmm_clf.fit(data['X_train'], data['y_train'])

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s")

    print("\n[3/4] Evaluating...")
    y_pred = hmm_clf.predict(data['X_test'])
    accuracy = accuracy_score(data['y_test'], y_pred)

    report = classification_report(
        data['y_test'], y_pred,
        target_names=data['class_names'], output_dict=True, zero_division=0
    )
    
    print(f"\n  HMM Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    print("\n[4/4] Saving model...")
    with open(os.path.join(output_dir, 'hmm_model.pkl'), 'wb') as f:
        pickle.dump(hmm_clf, f)

    results = {
        'model': 'HMM (Hidden Markov Model)',
        'accuracy': float(accuracy),
        'training_time_seconds': round(training_time, 1)
    }

    with open(os.path.join(output_dir, 'hmm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    train_hmm()
