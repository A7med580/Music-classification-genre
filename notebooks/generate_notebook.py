#!/usr/bin/env python3
"""Generate the training notebook as .ipynb"""
import json
import os

def md(source):
    return {"cell_type":"markdown","metadata":{},"source":[source]}

def code(source):
    return {"cell_type":"code","metadata":{},"source":[source],"outputs":[],"execution_count":None}

cells = []

# Title
cells.append(md("""# 🎵 Music Genre Classification & Emotion Detection
## Training Pipeline — Google Colab (T4 GPU)

**Two Tasks, One Dataset (GTZAN):**
1. **Music Genre Classification** — 10 genres
2. **Music Emotion Detection** — 4 emotions (mapped from genres)

**Features:** MFCC · Mel-Spectrogram · Chroma

**Models:** 1D-CNN · CRNN (CNN+BiLSTM) · SVM · GMM"""))

# Setup
cells.append(md("## 1️⃣ Setup & Dependencies"))
cells.append(code("""!pip install -q librosa scikit-learn matplotlib seaborn hmmlearn joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import pickle
import json
import os
import time
import warnings
import joblib
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print("✅ All dependencies loaded!")"""))

# Data Loading
cells.append(md("""## 2️⃣ Data Loading
Upload your `features_3_sec.csv` file when prompted."""))
cells.append(code("""from google.colab import files

# Upload CSV
print("📁 Upload features_3_sec.csv:")
uploaded = files.upload()
csv_name = list(uploaded.keys())[0]
df = pd.read_csv(csv_name)
print(f"\\n✅ Loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Genres: {sorted(df['label'].unique())}")
print(f"\\nSamples per genre:")
print(df['label'].value_counts().sort_index())"""))

# Feature Selection
cells.append(md("""## 3️⃣ Feature Engineering
Selecting **3 feature groups**: MFCC (40), Chroma (2), Mel-Spectrogram/Spectral (6) = **48 features**"""))
cells.append(code("""# Define feature groups
MFCC_COLS = [f'mfcc{i}_{s}' for i in range(1, 21) for s in ['mean', 'var']]
CHROMA_COLS = ['chroma_stft_mean', 'chroma_stft_var']
MELSPEC_COLS = ['spectral_centroid_mean', 'spectral_centroid_var',
                'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                'rolloff_mean', 'rolloff_var']

FEATURE_COLS = MFCC_COLS + CHROMA_COLS + MELSPEC_COLS
print(f"Total features: {len(FEATURE_COLS)}")
print(f"  MFCC: {len(MFCC_COLS)} features")
print(f"  Chroma: {len(CHROMA_COLS)} features")
print(f"  Mel-Spectrogram (spectral): {len(MELSPEC_COLS)} features")

X = df[FEATURE_COLS].values
y_genre = df['label'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"\\n✅ Features extracted and scaled: {X_scaled.shape}")"""))

# Emotion Mapping
cells.append(code("""# Genre → Emotion mapping (Russell's Valence-Arousal model)
GENRE_TO_EMOTION = {
    'disco': 'happy', 'pop': 'happy', 'reggae': 'happy',
    'blues': 'sad', 'country': 'sad',
    'metal': 'angry', 'rock': 'angry', 'hiphop': 'angry',
    'classical': 'calm', 'jazz': 'calm'
}

y_emotion = np.array([GENRE_TO_EMOTION[g] for g in y_genre])
EMOTION_LABELS = sorted(list(set(y_emotion)))

print("Emotion mapping:")
for emo in EMOTION_LABELS:
    genres = [g for g, e in GENRE_TO_EMOTION.items() if e == emo]
    count = np.sum(y_emotion == emo)
    print(f"  {emo}: {genres} → {count} samples")

# Encode labels
genre_encoder = LabelEncoder()
y_genre_enc = genre_encoder.fit_transform(y_genre)

emotion_encoder = LabelEncoder()
y_emotion_enc = emotion_encoder.fit_transform(y_emotion)

print(f"\\n✅ Genre classes: {list(genre_encoder.classes_)}")
print(f"✅ Emotion classes: {list(emotion_encoder.classes_)}")"""))

# Data Split
cells.append(code("""# Split data for both tasks (SAME split for fair comparison)
SEED = 42

# Genre split
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_scaled, y_genre_enc, test_size=0.2, random_state=SEED, stratify=y_genre_enc)

# Emotion split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_scaled, y_emotion_enc, test_size=0.2, random_state=SEED, stratify=y_emotion_enc)

# Reshape for CNN (add channel dim)
X_train_g_cnn = X_train_g.reshape(-1, X_train_g.shape[1], 1)
X_test_g_cnn = X_test_g.reshape(-1, X_test_g.shape[1], 1)
X_train_e_cnn = X_train_e.reshape(-1, X_train_e.shape[1], 1)
X_test_e_cnn = X_test_e.reshape(-1, X_test_e.shape[1], 1)

# Class weights for emotion (imbalanced)
from sklearn.utils.class_weight import compute_class_weight
emotion_weights = compute_class_weight('balanced', classes=np.unique(y_train_e), y=y_train_e)
emotion_class_weights = dict(enumerate(emotion_weights))

print(f"Genre  — Train: {X_train_g.shape[0]}, Test: {X_test_g.shape[0]}")
print(f"Emotion — Train: {X_train_e.shape[0]}, Test: {X_test_e.shape[0]}")
print(f"Emotion class weights: {emotion_class_weights}")"""))

# Model builders
cells.append(md("## 4️⃣ Model Definitions"))
cells.append(code("""def build_cnn(input_shape, num_classes, name='CNN'):
    \"\"\"1D-CNN with BatchNorm and Dropout\"\"\"
    model = keras.Sequential([
        layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name=name)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_crnn(input_shape, num_classes, name='CRNN'):
    \"\"\"CNN + Bidirectional LSTM\"\"\"
    model = keras.Sequential([
        layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name=name)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class GMMClassifier:
    \"\"\"One GMM per class — predict via max log-likelihood\"\"\"
    def __init__(self, n_components=16, cov_type='diag'):
        self.n_components = n_components
        self.cov_type = cov_type
        self.gmms = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(np.unique(y))
        for cls in self.classes:
            X_c = X[y == cls]
            n = min(self.n_components, len(X_c) // 2)
            n = max(n, 2)
            gmm = GaussianMixture(n_components=n, covariance_type=self.cov_type,
                                  max_iter=200, n_init=3, random_state=42)
            gmm.fit(X_c)
            self.gmms[cls] = gmm

    def predict(self, X):
        scores = np.array([self.gmms[c].score_samples(X) for c in self.classes]).T
        return np.array([self.classes[i] for i in np.argmax(scores, axis=1)])


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]


def plot_history(history, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
    a1.plot(history.history['accuracy'], label='Train')
    a1.plot(history.history['val_accuracy'], label='Val')
    a1.set_title(f'{title} — Accuracy'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history.history['loss'], label='Train')
    a2.plot(history.history['val_loss'], label='Val')
    a2.set_title(f'{title} — Loss'); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()

print("✅ All model builders defined!")"""))

# GENRE TRAINING
cells.append(md("## 5️⃣ Task 1: Music Genre Classification"))

# Genre CNN
cells.append(code("""print("="*60)
print("TRAINING 1D-CNN FOR GENRE CLASSIFICATION")
print("="*60)

genre_cnn = build_cnn((X_train_g_cnn.shape[1], 1), len(genre_encoder.classes_), 'Genre_CNN')
genre_cnn.summary()

t0 = time.time()
h_gcnn = genre_cnn.fit(X_train_g_cnn, y_train_g,
                       validation_data=(X_test_g_cnn, y_test_g),
                       epochs=100, batch_size=32,
                       callbacks=get_callbacks(), verbose=1)
gcnn_time = time.time() - t0

y_pred_gcnn = np.argmax(genre_cnn.predict(X_test_g_cnn), axis=1)
gcnn_acc = accuracy_score(y_test_g, y_pred_gcnn)
print(f"\\n🎯 Genre CNN Accuracy: {gcnn_acc*100:.2f}%")
print(classification_report(y_test_g, y_pred_gcnn, target_names=genre_encoder.classes_))
plot_history(h_gcnn, 'Genre CNN')
plot_cm(y_test_g, y_pred_gcnn, genre_encoder.classes_, 'Genre CNN Confusion Matrix')"""))

# Genre CRNN
cells.append(code("""print("="*60)
print("TRAINING CRNN FOR GENRE CLASSIFICATION")
print("="*60)

genre_crnn = build_crnn((X_train_g_cnn.shape[1], 1), len(genre_encoder.classes_), 'Genre_CRNN')
genre_crnn.summary()

t0 = time.time()
h_gcrnn = genre_crnn.fit(X_train_g_cnn, y_train_g,
                         validation_data=(X_test_g_cnn, y_test_g),
                         epochs=100, batch_size=32,
                         callbacks=get_callbacks(), verbose=1)
gcrnn_time = time.time() - t0

y_pred_gcrnn = np.argmax(genre_crnn.predict(X_test_g_cnn), axis=1)
gcrnn_acc = accuracy_score(y_test_g, y_pred_gcrnn)
print(f"\\n🎯 Genre CRNN Accuracy: {gcrnn_acc*100:.2f}%")
print(classification_report(y_test_g, y_pred_gcrnn, target_names=genre_encoder.classes_))
plot_history(h_gcrnn, 'Genre CRNN')
plot_cm(y_test_g, y_pred_gcrnn, genre_encoder.classes_, 'Genre CRNN Confusion Matrix')"""))

# Genre SVM + GMM
cells.append(code("""# SVM
print("="*60)
print("TRAINING SVM FOR GENRE CLASSIFICATION")
print("="*60)
t0 = time.time()
genre_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
genre_svm.fit(X_train_g, y_train_g)
gsvm_time = time.time() - t0
y_pred_gsvm = genre_svm.predict(X_test_g)
gsvm_acc = accuracy_score(y_test_g, y_pred_gsvm)
print(f"🎯 Genre SVM Accuracy: {gsvm_acc*100:.2f}% (Time: {gsvm_time:.1f}s)")
print(classification_report(y_test_g, y_pred_gsvm, target_names=genre_encoder.classes_))
plot_cm(y_test_g, y_pred_gsvm, genre_encoder.classes_, 'Genre SVM Confusion Matrix')

# GMM
print("\\n" + "="*60)
print("TRAINING GMM FOR GENRE CLASSIFICATION")
print("="*60)
t0 = time.time()
genre_gmm = GMMClassifier(n_components=16, cov_type='diag')
genre_gmm.fit(X_train_g, y_train_g)
ggmm_time = time.time() - t0
y_pred_ggmm = genre_gmm.predict(X_test_g)
ggmm_acc = accuracy_score(y_test_g, y_pred_ggmm)
print(f"🎯 Genre GMM Accuracy: {ggmm_acc*100:.2f}% (Time: {ggmm_time:.1f}s)")
print(classification_report(y_test_g, y_pred_ggmm, target_names=genre_encoder.classes_))
plot_cm(y_test_g, y_pred_ggmm, genre_encoder.classes_, 'Genre GMM Confusion Matrix')"""))

# Genre Summary
cells.append(code("""genre_results = {
    'CNN': {'accuracy': gcnn_acc, 'time': gcnn_time},
    'CRNN': {'accuracy': gcrnn_acc, 'time': gcrnn_time},
    'SVM': {'accuracy': gsvm_acc, 'time': gsvm_time},
    'GMM': {'accuracy': ggmm_acc, 'time': ggmm_time},
}

print("\\n" + "="*60)
print("📊 GENRE CLASSIFICATION RESULTS")
print("="*60)
print(f"{'Model':<10} {'Accuracy':>10} {'Time':>10}")
print("-"*35)
for m, r in sorted(genre_results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"{m:<10} {r['accuracy']*100:>9.2f}% {r['time']:>8.1f}s")"""))

# EMOTION TRAINING
cells.append(md("## 6️⃣ Task 2: Music Emotion Detection"))

# Emotion CNN
cells.append(code("""print("="*60)
print("TRAINING 1D-CNN FOR EMOTION DETECTION")
print("="*60)

n_emo = len(emotion_encoder.classes_)
emotion_cnn = build_cnn((X_train_e_cnn.shape[1], 1), n_emo, 'Emotion_CNN')

t0 = time.time()
h_ecnn = emotion_cnn.fit(X_train_e_cnn, y_train_e,
                         validation_data=(X_test_e_cnn, y_test_e),
                         epochs=100, batch_size=32,
                         class_weight=emotion_class_weights,
                         callbacks=get_callbacks(), verbose=1)
ecnn_time = time.time() - t0

y_pred_ecnn = np.argmax(emotion_cnn.predict(X_test_e_cnn), axis=1)
ecnn_acc = accuracy_score(y_test_e, y_pred_ecnn)
print(f"\\n🎯 Emotion CNN Accuracy: {ecnn_acc*100:.2f}%")
print(classification_report(y_test_e, y_pred_ecnn, target_names=emotion_encoder.classes_))
plot_history(h_ecnn, 'Emotion CNN')
plot_cm(y_test_e, y_pred_ecnn, emotion_encoder.classes_, 'Emotion CNN Confusion Matrix')"""))

# Emotion CRNN
cells.append(code("""print("="*60)
print("TRAINING CRNN FOR EMOTION DETECTION")
print("="*60)

emotion_crnn = build_crnn((X_train_e_cnn.shape[1], 1), n_emo, 'Emotion_CRNN')

t0 = time.time()
h_ecrnn = emotion_crnn.fit(X_train_e_cnn, y_train_e,
                           validation_data=(X_test_e_cnn, y_test_e),
                           epochs=100, batch_size=32,
                           class_weight=emotion_class_weights,
                           callbacks=get_callbacks(), verbose=1)
ecrnn_time = time.time() - t0

y_pred_ecrnn = np.argmax(emotion_crnn.predict(X_test_e_cnn), axis=1)
ecrnn_acc = accuracy_score(y_test_e, y_pred_ecrnn)
print(f"\\n🎯 Emotion CRNN Accuracy: {ecrnn_acc*100:.2f}%")
print(classification_report(y_test_e, y_pred_ecrnn, target_names=emotion_encoder.classes_))
plot_history(h_ecrnn, 'Emotion CRNN')
plot_cm(y_test_e, y_pred_ecrnn, emotion_encoder.classes_, 'Emotion CRNN Confusion Matrix')"""))

# Emotion SVM + GMM
cells.append(code("""# SVM
print("="*60)
print("TRAINING SVM FOR EMOTION DETECTION")
print("="*60)
t0 = time.time()
emotion_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
emotion_svm.fit(X_train_e, y_train_e)
esvm_time = time.time() - t0
y_pred_esvm = emotion_svm.predict(X_test_e)
esvm_acc = accuracy_score(y_test_e, y_pred_esvm)
print(f"🎯 Emotion SVM Accuracy: {esvm_acc*100:.2f}%")
print(classification_report(y_test_e, y_pred_esvm, target_names=emotion_encoder.classes_))
plot_cm(y_test_e, y_pred_esvm, emotion_encoder.classes_, 'Emotion SVM Confusion Matrix')

# GMM
print("\\n" + "="*60)
print("TRAINING GMM FOR EMOTION DETECTION")
print("="*60)
t0 = time.time()
emotion_gmm = GMMClassifier(n_components=16, cov_type='diag')
emotion_gmm.fit(X_train_e, y_train_e)
egmm_time = time.time() - t0
y_pred_egmm = emotion_gmm.predict(X_test_e)
egmm_acc = accuracy_score(y_test_e, y_pred_egmm)
print(f"🎯 Emotion GMM Accuracy: {egmm_acc*100:.2f}%")
print(classification_report(y_test_e, y_pred_egmm, target_names=emotion_encoder.classes_))
plot_cm(y_test_e, y_pred_egmm, emotion_encoder.classes_, 'Emotion GMM Confusion Matrix')"""))

# Emotion Summary
cells.append(code("""emotion_results = {
    'CNN': {'accuracy': ecnn_acc, 'time': ecnn_time},
    'CRNN': {'accuracy': ecrnn_acc, 'time': ecrnn_time},
    'SVM': {'accuracy': esvm_acc, 'time': esvm_time},
    'GMM': {'accuracy': egmm_acc, 'time': egmm_time},
}

print("\\n" + "="*60)
print("📊 EMOTION DETECTION RESULTS")
print("="*60)
print(f"{'Model':<10} {'Accuracy':>10} {'Time':>10}")
print("-"*35)
for m, r in sorted(emotion_results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"{m:<10} {r['accuracy']*100:>9.2f}% {r['time']:>8.1f}s")"""))

# Cross-task comparison
cells.append(md("## 7️⃣ Cross-Task Comparison"))
cells.append(code("""fig, ax = plt.subplots(figsize=(12, 6))
models = ['CNN', 'CRNN', 'SVM', 'GMM']
x = np.arange(len(models))
w = 0.35

g_vals = [genre_results[m]['accuracy'] for m in models]
e_vals = [emotion_results[m]['accuracy'] for m in models]

b1 = ax.bar(x - w/2, g_vals, w, label='Genre Classification', color='#2196F3', alpha=0.85)
b2 = ax.bar(x + w/2, e_vals, w, label='Emotion Detection', color='#FF9800', alpha=0.85)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.1%}',
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Cross-Task Comparison: Genre vs Emotion', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Save models
cells.append(md("## 8️⃣ Save All Models"))
cells.append(code("""# Create output directory
os.makedirs('saved_models', exist_ok=True)

# Save Keras models
genre_cnn.save('saved_models/genre_cnn_model.keras')
genre_crnn.save('saved_models/genre_crnn_model.keras')
emotion_cnn.save('saved_models/emotion_cnn_model.keras')
emotion_crnn.save('saved_models/emotion_crnn_model.keras')

# Save sklearn/custom models
joblib.dump(genre_svm, 'saved_models/genre_svm_model.pkl')
joblib.dump(emotion_svm, 'saved_models/emotion_svm_model.pkl')
with open('saved_models/genre_gmm_model.pkl', 'wb') as f:
    pickle.dump(genre_gmm, f)
with open('saved_models/emotion_gmm_model.pkl', 'wb') as f:
    pickle.dump(emotion_gmm, f)

# Save encoders and scaler
joblib.dump(scaler, 'saved_models/feature_scaler.pkl')
joblib.dump(genre_encoder, 'saved_models/genre_label_encoder.pkl')
joblib.dump(emotion_encoder, 'saved_models/emotion_label_encoder.pkl')

# Save config
config = {
    'feature_columns': FEATURE_COLS,
    'mfcc_cols': MFCC_COLS,
    'chroma_cols': CHROMA_COLS,
    'melspec_cols': MELSPEC_COLS,
    'genre_to_emotion': GENRE_TO_EMOTION,
    'genre_classes': list(genre_encoder.classes_),
    'emotion_classes': list(emotion_encoder.classes_),
    'genre_results': {k: {'accuracy': v['accuracy'], 'time': round(v['time'], 1)} for k, v in genre_results.items()},
    'emotion_results': {k: {'accuracy': v['accuracy'], 'time': round(v['time'], 1)} for k, v in emotion_results.items()},
}
with open('saved_models/training_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ All models saved!")
print("\\nFiles in saved_models/:")
for f in sorted(os.listdir('saved_models')):
    size = os.path.getsize(f'saved_models/{f}')
    print(f"  {f} ({size/1024:.0f} KB)")"""))

# Download
cells.append(code("""# Zip and download
!cd saved_models && zip -r ../saved_models.zip .
files.download('saved_models.zip')
print("\\n🎉 Download started! Extract and place in your project's ml/ directory.")"""))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), 'train_models.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"✅ Notebook saved to: {out_path}")
