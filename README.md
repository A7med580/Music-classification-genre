# 🎵 AI Music Genre Classification
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.x-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-Node.js-339933.svg)](https://nodejs.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.9%25-brightgreen.svg)](https://github.com/A7med580/Music-classification-genre)

A professional, high-performance web application that classifies music genres using Deep Learning. This project features a React-based "Premium" interface, a Node.js orchestration layer, and a Python-powered Machine Learning engine utilizing **1D-CNN** and **GMM** architectures.

---

## 🎯 Project Performance

Our deep learning approach achieved state-of-the-art results on the GTZAN dataset, significantly outperforming traditional statistical baselines.

| Model | Accuracy | F1 (Macro) | Type | Status |
|-------|:--------:|:----------:|------|--------|
| **1D-CNN** | **91.9%** | **0.92** | Deep Learning | 🏆 Best |
| Random Forest | 87% | 0.87 | Ensemble | Legacy |
| GMM Baseline | 71.4% | 0.71 | Classical Statistical | Baseline |

> [!TIP]
> **Scientific Significance**: The CNN architecture demonstrates a **+20.5%** performance lift over the GMM baseline, proving the efficacy of hierarchical feature learning for complex audio temporal patterns.

---

## 🚀 System Architecture

### 1. Frontend (Premium Experience)
- **Tech Stack**: React + Vite + Vanilla CSS.
- **Design**: Premium Glassmorphism UI with dynamic animations.
- **Features**: Real-time track analysis, confidence heatmaps, and genre probability visualization.

### 2. Backend (Orchestration)
- **Tech Stack**: Node.js + Express.
- **Logic**: Manages secure file uploads, handles authentication, and orchestrates the Python ML lifecycle via specialized subprocess hooks.

### 3. Machine Learning (Core Engine)
- **Deep Learning**: 1D Convolutional Neural Network built with TensorFlow/Keras.
- **Baseline**: Gaussian Mixture Model (GMM) for statistical benchmarking.
- **Features**: Advanced extraction using `librosa` (20 MFCCs, Spectral Centroid, ZCR, Chroma, etc. — Total 57 dimensions).
- **Dataset**: GTZAN Genre Collection (1000 tracks, 10 genres).

---

## 🔬 Scientific Analysis

### Top Predictive Features
1. **MFCCs**: Captures the unique "timbre" of instruments (e.g., distorted guitars in Metal vs. clean pianos in Classical).
2. **Spectral Centroid**: Measures the "center of mass" of the sound — critical for distinguishing bright genres (Pop) from warm ones (Jazz).
3. **Chroma STFT**: Captures harmonic and melodic content, identifying the distinct chordal structures of Country and Blues.

### Feature Reliability Analysis
We performed MI (Mutual Information) scoring and ablation studies to identify redundant features like **Tempo** and **RMS Energy**, which showed high intra-class variance. Full details in [FEATURE_SELECTION.md](file:///Users/mohamedali/Desktop/Music-classification-genre/docs/FEATURE_SELECTION.md).

---

## 📁 Project Structure

```
Music-classification-genre/
├── client/              # React Frontend (Vite)
├── server/              # Node.js Backend API
├── ml/                  # Machine Learning Core
│   ├── core/            # Feature extraction & preprocessing
│   ├── models/          # 1D-CNN & GMM implementations
│   ├── outputs/         # Confusion matrices, ROC curves, plots
│   └── predict.py       # Production inference pipeline
├── docs/                # Technical documentation & Ablation studies
├── reports/             # 20-page Technical Report
├── presentation/        # Slide contents & Demo scripts
└── Data/                # Feature datasets (GTZAN)
```

---

## 🛠️ Installation & Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/A7med580/Music-classification-genre.git

# Install all dependencies (Frontend & Backend)
npm install

# Initialize Python Virtual Environment
chmod +x setup_env.sh && ./setup_env.sh
source venv/bin/activate
```

### 2. Training (Optional)
If you wish to retrain all models from scratch:
```bash
cd ml && python train_all_models.py
```

### 3. Launch Development Server
```bash
npm start
```
- **UI**: `http://localhost:5173`
- **API**: `http://localhost:3000`

---

## 📊 Documentation Hub

| Category | Links |
|----------|-------|
| **Technical Analysis** | [Technical Report](file:///Users/mohamedali/Desktop/Music-classification-genre/reports/TECHNICAL_REPORT.md) \| [Comparison Analysis](file:///Users/mohamedali/Desktop/Music-classification-genre/docs/MODEL_COMPARISON.md) |
| **Research** | [Feature Selection](file:///Users/mohamedali/Desktop/Music-classification-genre/docs/FEATURE_SELECTION.md) \| [Cross-Study (SER)](file:///Users/mohamedali/Desktop/Music-classification-genre/docs/CROSS_STUDY_MODELING.md) |
| **Deliverables** | [Slide Content](file:///Users/mohamedali/Desktop/Music-classification-genre/presentation/slides_content.md) \| [Demo Script](file:///Users/mohamedali/Desktop/Music-classification-genre/presentation/demo_script.txt) |

---

## 🔧 Troubleshooting

- **Buffer Issues**: If large audio files fail, ensure the `uploads/` directory has write permissions.
- **Python Path**: If the backend can't find Python, update the path in `server/index.js` or ensure `venv` is active.
- **TF-CPU**: On low-RAM systems, use `pip install tensorflow-cpu` to avoid memory overhead.

---
© 2026 AI Music Research Group | [May 1st Graduation Deadline]
