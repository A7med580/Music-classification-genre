# 🎵 SonicSense AI: Dual-Task Music Genre & Emotion Analysis
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://tensorflow.org/)
[![React 18](https://img.shields.io/badge/React-18.x-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SonicSense AI is a production-grade, high-performance web platform that performs simultaneous **Music Genre Classification** and **Emotion Detection**. Powered by a multi-architecture AI engine (CNN, CRNN, SVM, GMM, HMM), it transforms raw audio into deep emotional and stylistic insights with state-of-the-art accuracy.

---

## 🚀 Key Features

*   **Dual-Task Inference**: Single upload analysis for both Genre (10 classes) and Emotion (Happy, Sad, Angry, Calm).
*   **Multi-Model Intelligence**: Choose between Deep Learning (CNN/CRNN) or Statistical Baselines (SVM/GMM/HMM) via the UI.
*   **Premium Analytics Dashboard**: High-fidelity interface with dual-panel results, confidence bars, and dynamic emotion-based color themes.
*   **Optimized Feature Pipeline**: Advanced 48-dimension feature vector (MFCCs, Mel-Spectrogram, Chroma, Spectral Contrast).
*   **Colab Integration**: Modular training workflow optimized for T4 GPU environments.

---

## 🎯 Model Performance Matrix

| Task | Best Model | Accuracy | F1-Score | Type |
| :--- | :--- | :--- | :--- | :--- |
| **Music Genre** | **1D-CNN** | **90.3%** | **0.90** | Deep Learning |
| **Emotion Detection** | **1D-CNN** | **92.1%** | **0.92** | Deep Learning |
| Music Genre | CRNN | 88.5% | 0.88 | Hybrid DL |
| Emotion Detection | SVM | 82.7% | 0.82 | Statistical |

> [!IMPORTANT]
> **Production Note**: The 1D-CNN architecture currently provides the most reliable balance between inference speed and accuracy for both classification tasks.

---

## 🛠️ System Architecture

### 1. Frontend (The Experience)
- **Glassmorphism UI**: A sleek, dark-themed dashboard built with React.
- **Dynamic Feedback**: Real-time analysis states and model-specific performance metrics.
- **Accessibility**: Direct dashboard access (authentication-free) for instant research use.

### 2. Backend (The Orchestrator)
- **Node.js/Express**: High-concurrency server managing file lifecycle and ML orchestration.
- **Python-Link**: Seamless subprocess integration with the Python inference engine.

### 3. ML Engine (The Core)
- **Feature Extraction**: Powered by `librosa`, extracting MFCCs, Chroma, and Spectral features.
- **Versatile Architectures**: Supports Deep Learning (Keras/TensorFlow) and Classical ML (Scikit-Learn/Joblib).
- **Data Consistency**: Built-in feature scaling and label encoding synchronization.

---

## 📁 Folder Structure

```text
Music-classification-genre/
├── client/              # React Dashboard (Vite)
├── server/              # Express API Orchestrator
├── ml/                  # Machine Learning Core
│   ├── models/          # Trained model artifacts (.keras, .pkl)
│   ├── core/            # Feature engineering (librosa)
│   └── predict.py       # Dual-task inference pipeline
├── docs/                # Consolidated Research & Reports
│   ├── presentation/    # Slides and demo material
│   ├── reports/         # 20-page Technical Report
│   └── research/        # Feature selection & ablation studies
└── notebooks/           # Colab training notebooks & setup scripts
```

---

## ⚙️ Installation & Setup

### 1. Environment Preparation
```bash
# Clone the repository
git clone https://github.com/A7med580/Music-classification-genre.git

# Install Core & UI Dependencies
npm install
cd client && npm install && cd ..
```

### 2. Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ml/requirements.txt
```

### 3. Launch Development Suite
```bash
npm start
```
*   **Dashboard**: `http://localhost:5173`
*   **API**: `http://localhost:3000`

---

## 🔬 Scientific Methodology

The pipeline utilizes a **Late Fusion** inspired approach where a shared feature extractor feeds into task-specific heads. 
- **Time-Domain**: Zero Crossing Rate (ZCR) and RMS Energy.
- **Frequency-Domain**: MFCCs (20 coefficients) and Spectral Centroid.
- **Harmonic-Domain**: Chroma STFT for chordal structure analysis.

Full scientific analysis and ablation studies are available in the [docs/research](docs/research) directory.

---

© 2026 AI Music Research Group | **SonicSense AI Platform**
