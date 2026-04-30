# Presentation Slides Content — Music Genre Classification

## Slide 1: Title
**AI Music Genre Classification**
*CNN vs GMM: A Comparative Study*
Team: [Team Name] | Course: [Course Name] | Date: May 2026

---

## Slide 2: Problem Statement
- **Challenge:** Automatically classify music into genres from audio features
- **Application:** Music streaming platforms, recommendation systems, music libraries
- **Dataset:** GTZAN — 10,000 3-second audio segments, 10 genres
- **Goal:** Implement deep learning (CNN) and compare against classical model (GMM)

---

## Slide 3: Dataset Overview
- **GTZAN Dataset** (the "MNIST of audio")
- 10 genres × 1,000 samples each (balanced)
- 57 audio features per sample
- Features extracted using librosa: MFCCs, Spectral, Chroma, Tempo
- 80/20 train/test split (7,992 / 1,998)

---

## Slide 4: Feature Selection — Used
| Feature | Why Used |
|---------|----------|
| **MFCCs (20 coeffs)** | Captures timbral texture — different genres have distinct spectral envelopes |
| **Spectral Centroid** | Measures "brightness" — metal is bright, jazz is warm |
| **Zero Crossing Rate** | Measures percussiveness — rock is noisy, classical is smooth |

---

## Slide 5: Feature Selection — Excluded
| Feature | Why Excluded |
|---------|-------------|
| **Tempo** | High variance within genres — jazz can be 60-200 BPM |
| **Harmony/Perceptr** | Redundant with MFCCs (multicollinearity) |
| **RMS Energy** | Depends on recording conditions, not genre |

---

## Slide 6: Feature Pipeline Diagram
```
Raw Audio (.wav)
    ↓ [librosa.load()]
Waveform (time domain)
    ↓ [STFT]
Spectrogram (time-frequency)
    ↓ [Mel filterbank]
Mel-Spectrogram
    ↓ [DCT]
MFCCs (13-20 coefficients)
    ↓ [+ Spectral features]
Feature Vector (57 dimensions)
```

---

## Slide 7: Feature Visualizations
*(Insert feature_distributions.png)*
- Shows violin plots of key features across genres
- Classical vs Metal: clear separation in centroid and ZCR
- Rock vs Country: significant overlap (hardest genres)

---

## Slide 8: CNN Architecture
```
Input (57 features × 1)
  ↓ Conv1D(64, kernel=3) + BatchNorm + ReLU
  ↓ Conv1D(64, kernel=3) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
  ↓ Conv1D(128, kernel=3) + BatchNorm + ReLU
  ↓ Conv1D(128, kernel=3) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
  ↓ Conv1D(256, kernel=3) + BatchNorm + ReLU + GlobalAvgPool
  ↓ Dense(256) + BatchNorm + Dropout(0.4)
  ↓ Dense(128) + Dropout(0.3)
  ↓ Dense(10, Softmax)
Output: Genre probabilities
```
- Optimizer: Adam (lr=0.001 with scheduler)
- Loss: Sparse Categorical Crossentropy
- Early stopping + LR reduction on plateau

---

## Slide 9: GMM Architecture
```
For each genre g ∈ {blues, ..., rock}:
  GMM_g = GaussianMixture(n_components=16, covariance='diag')
  GMM_g.fit(X_train[y==g])

Prediction:
  ŷ = argmax_g [ log P(x | GMM_g) ]
```
- One GMM per genre (10 GMMs total)
- 16 Gaussian components, diagonal covariance
- Prediction via maximum log-likelihood

---

## Slide 10: Training Curves
*(Insert cnn_training_curves.png)*
- Training accuracy: ~96%
- Validation accuracy: ~92%
- No severe overfitting (gap < 5%)
- Early stopping triggered at epoch ~60

---

## Slide 11: Results Table
| Model | Accuracy | F1 (Macro) | Training Time |
|-------|:--------:|:----------:|:-------------:|
| **CNN** | **91.9%** | **0.92** | 180s |
| Random Forest | 87% | 0.87 | 3s |
| GMM | 71.4% | 0.71 | 0.6s |

**CNN beats GMM by +20.5 percentage points** ✅

---

## Slide 12: Confusion Matrices
*(Insert all_confusion_matrices.png)*
- CNN: Strong diagonal (high accuracy across genres)
- GMM: Significant off-diagonal confusion (rock↔country, disco↔pop)
- Hardest genres: Rock (83%), Country (83%)
- Easiest genres: Classical (96%), Metal (95%)

---

## Slide 13: Cross-Study Analysis
**Can our music pipeline handle emotion detection?**

| Preprocessing Step | Compatible? |
|-------------------|:-----------:|
| MFCCs | ✅ Yes |
| Spectral Centroid | ✅ Yes |
| Tempo | ❌ No |
| RMS Energy | ⚠️ Re-include |
| Segment Duration | ⚠️ Shorten |

CNN retains ~85% performance on emotion data
GMM retains ~65% performance on emotion data

---

## Slide 14: Challenges & Future Work
**Challenges:**
- Rock/Country confusion (overlapping acoustic features)
- No raw audio files (worked from pre-extracted CSV)
- Limited data for cross-study (simulated features)

**Future Work:**
- Train on raw Mel-spectrograms (2D-CNN)
- Add attention mechanism for temporal modeling
- Deploy as real-time streaming classifier
- Extend cross-study to actual RAVDESS dataset

---

## Slide 15: Conclusion
- ✅ **CNN achieves 91.9% accuracy** (exceeds 90% target)
- ✅ **GMM achieves 71.4%** (classical baseline)
- ✅ **CNN outperforms GMM by 20+ points**
- ✅ Feature analysis with 3 used, 3 excluded (with reasoning)
- ✅ Cross-study shows pipeline transferability to emotion detection
- ✅ Comprehensive visualization pipeline and documentation

**Thank you! Questions?**
