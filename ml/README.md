# 🧠 Machine Learning Core
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/librosa-0.x-blue.svg)](https://librosa.org/)

The analytical heart of the Music Genre Classification system. This directory contains the architectures, training pipelines, and evaluation metrics for our multi-model approach.

## 🏗️ Model Architectures

### 1. 1D-CNN (Deep Learning) - **91.9% Accuracy**
- **Architecture**: multi-layer 1D Convolutional Neural Network with Dropout and Batch Normalization.
- **Optimization**: Adam optimizer with categorical cross-entropy loss.
- **Why it wins**: Capable of learning hierarchical temporal features from raw MFCC sequences.

### 2. GMM Baseline (Classical) - **71.4% Accuracy**
- **Type**: Gaussian Mixture Model utilizing diagonal covariance.
- **Role**: Serves as a statistical baseline to demonstrate the performance gap between classical and deep learning approaches.

### 3. Random Forest (Legacy) - **87% Accuracy**
- **Type**: Ensemble Classifier with 100 estimators.
- **Role**: Stable legacy model currently used in production for fast inference.

## 📊 Feature Engineering Pipeline

We extract 57 features per track using `librosa`:
- **MFCCs (20)**: Mel-Frequency Cepstral Coefficients (Timbral texture).
- **Spectral Centroid**: Brightness.
- **Spectral Roll-off**: Signal shaping.
- **Zero Crossing Rate**: Perceptual noise/attack.
- **Chroma & Harmony**: Melodic and harmonic profiles.

## 📁 Repository Map

- `models/`: High-level model definitions (`cnn_model.py`, `gmm_baseline.py`).
- `core/`: Low-level data handlers (`preprocessing.py`) and feature extractors (`features.py`).
- `outputs/`: Visualization artifacts including training history, confusion matrices, and ROC curves.
- `train_all_models.py`: Orchestration script to retrain the entire suite.
- `predict.py`: The production entry point for single-track inference.

## 🧪 Evaluation

Our models are evaluated using:
- **Accuracy & F1-Score**: Balance between precision and recall.
- **Ablation Studies**: Testing feature subsets to find the most predictive dimensions.
- **Cross-Study Validation**: Assessing pipeline compatibility with SER (Speech Emotion Recognition) datasets.

---
© 2026 AI Music Research Group
