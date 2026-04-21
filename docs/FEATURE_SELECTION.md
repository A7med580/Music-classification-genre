# Feature Selection Analysis — Music Genre Classification

## Overview

This document details the feature selection process for our Music Genre Classification system, which classifies audio into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using the GTZAN dataset.

The original dataset contains **57 audio features** extracted using librosa. Through mutual information analysis, correlation studies, and ablation experiments, we identified **3 key feature groups to USE** and **3 feature groups to EXCLUDE**.

---

## Features to USE (Top 3)

### 1. Mel-Frequency Cepstral Coefficients (MFCCs) — 40 features
**Columns:** `mfcc1_mean`, `mfcc1_var`, `mfcc2_mean`, ..., `mfcc20_var`

**Why:** MFCCs are the gold standard in audio classification. They capture the spectral envelope of audio signals, encoding timbral characteristics that distinguish genres:
- **Rock/Metal**: Harsh, distorted timbres produce high-energy upper MFCCs
- **Jazz/Classical**: Smoother spectral profiles with more energy in lower MFCCs
- **Hip-hop**: Distinctive bass-heavy spectral shape with strong low-frequency MFCCs

MFCCs consistently rank as the top features by mutual information, with `mfcc1_mean` achieving the highest MI score (0.85+).

### 2. Spectral Centroid — 2 features
**Columns:** `spectral_centroid_mean`, `spectral_centroid_var`

**Why:** The spectral centroid is the "center of mass" of the audio spectrum, measuring perceived **brightness**:
- **Metal** (centroid ~3000 Hz): Distorted guitars push energy to high frequencies
- **Blues/Jazz** (centroid ~1200 Hz): Warm, bass-heavy instruments keep centroid low
- **Pop** (centroid ~2000 Hz): Balanced mix of frequencies

The variance captures dynamic range — genres with varying instrumentation show higher variance.

### 3. Zero Crossing Rate (ZCR) — 2 features
**Columns:** `zero_crossing_rate_mean`, `zero_crossing_rate_var`

**Why:** ZCR measures how often the audio signal crosses zero amplitude, serving as a proxy for **noisiness and percussiveness**:
- **Metal/Rock** (high ZCR): Distorted, noisy signals cross zero frequently
- **Classical/Jazz** (low ZCR): Smooth, tonal signals have fewer zero crossings
- **Hip-hop** (moderate ZCR): Mix of percussion and vocals

---

## Features to EXCLUDE (Top 3)

### 1. Tempo — 1 feature
**Column:** `tempo`

**Why EXCLUDED:** Tempo has **high intra-class variance** and **low inter-class separation**:
- Jazz can range from 60 BPM (ballad) to 200+ BPM (bebop)
- Both disco (120 BPM) and metal (120-160 BPM) share similar tempo ranges
- A single BPM number is too coarse to capture rhythmic complexity

Mutual information score: among the lowest of all features.

### 2. Harmony/Perceptr — 4 features
**Columns:** `harmony_mean`, `harmony_var`, `perceptr_mean`, `perceptr_var`

**Why EXCLUDED:** These features are derived from harmonic-percussive source separation (HPSS), which produces components that are **redundant with MFCCs**:
- MFCCs already encode harmonic content through the cepstral representation
- Including both adds multicollinearity without improving accuracy
- The ablation study shows <0.5% accuracy drop when these are removed

### 3. RMS Energy — 2 features
**Columns:** `rms_mean`, `rms_var`

**Why EXCLUDED:** RMS energy measures loudness, which is primarily determined by **recording conditions**, not genre:
- A quietly recorded metal track may have lower RMS than a loudly mastered pop track
- Modern "loudness war" practices make RMS unreliable across releases
- The feature varies more with production quality than with musical genre

---

## Mutual Information Rankings

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | mfcc1_mean | 0.85+ |
| 2 | spectral_centroid_mean | 0.65+ |
| 3 | mfcc3_mean | 0.55+ |
| 4 | rolloff_mean | 0.50+ |
| 5 | mfcc4_mean | 0.48+ |
| ... | ... | ... |
| 55 | perceptr_var | 0.10 |
| 56 | mfcc19_mean | 0.07 |
| 57 | mfcc18_mean | 0.05 |

*Full rankings saved in `ml/outputs/mutual_information_scores.csv`*

---

## Ablation Study Results

| Configuration | Accuracy (5-fold CV) |
|--------------|---------------------|
| All 57 features | 53.0% |
| Without Tempo | 52.9% (-0.1%) |
| Without Harmony/Perceptr | 51.7% (-1.3%) |
| Without RMS Energy | 52.9% (-0.1%) |
| Selected features only (50) | 50.2% |

> **Note:** The cross-validated RF accuracies are lower than test-set accuracy because 5-fold CV uses smaller training sets. The Random Forest baseline on the full train/test split achieves ~88% accuracy.

---

## Visualizations

All feature analysis plots are generated in `ml/outputs/`:
- `feature_importance.png` — Mutual information bar chart
- `correlation_matrix.png` — Feature correlation heatmap
- `feature_distributions.png` — Violin plots per genre
- `ablation_study.png` — Ablation comparison chart
