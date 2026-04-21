# Cross-Study Analysis: Preprocessing Compatibility

## Music Genre Classification → Speech Emotion Recognition (SER)

This document evaluates whether the preprocessing pipeline developed for music genre classification can transfer to a speech emotion recognition task. We analyze each preprocessing step for compatibility.

---

## Preprocessing Pipeline Comparison

| Step | Music Genre | Emotion Detection | Compatible? | Explanation |
|------|-------------|-------------------|:-----------:|-------------|
| **Sampling Rate** | 22,050 Hz | 22,050 Hz (or 16,000 Hz) | ✅ Yes | Standard audio sampling rate works for both domains. Some SER datasets use 16 kHz, but upsampling to 22,050 Hz is trivial and doesn't lose information. |
| **Segment Duration** | 3 seconds | 1–3 seconds | ⚠️ Partial | Music segments at 3s capture enough temporal context for genre patterns. Emotional utterances are typically 1–5 seconds. For short utterances (<2s), shorter analysis windows may be needed to avoid zero-padding artifacts. |
| **MFCC Extraction** | 20 coefficients | 20 coefficients | ✅ Yes | MFCCs capture the spectral envelope, encoding timbral information useful for both tasks. In music, they encode instrument timbre; in speech, they encode vocal quality and emotional prosody. This is the **most transferable feature**. |
| **Spectral Centroid** | Measures brightness of instruments | Correlates with vocal intensity | ✅ Yes | Angry speech has higher spectral centroids (sharper harmonics), while sad speech has lower centroids — the same discriminative pattern observed between metal (high) and jazz (low) in music. |
| **Zero Crossing Rate** | Percussive vs smooth distinction | Speech rate and energy correlation | ✅ Yes | Energetic speech (anger, happiness) has higher ZCR, analogous to how percussive genres (rock, metal) have higher ZCR than tonal genres (classical, jazz). |
| **Spectral Bandwidth** | Spectral spread of instrumentation | Vocal spectral richness | ✅ Yes | Both domains benefit from measuring spectral spread. Emotionally expressive speech widens spectral bandwidth, similar to how complex instrumentation broadens the spectrum. |
| **Spectral Rolloff** | High-frequency energy boundary | Vocal energy distribution | ✅ Yes | Indicates where most spectral energy is concentrated. Compatible across domains without modification. |
| **Chroma STFT** | Pitch class energy distribution | Less useful for emotion | ⚠️ Partial | Chroma captures harmonic content and key signatures, which are more relevant for music than emotion. For SER, pitch contour features would be more appropriate, but chroma can still capture some prosodic patterns. |
| **Tempo** | Beat estimation (**EXCLUDED**) | Not applicable | ❌ No | Speech does not have a musical tempo. The beat tracker produces meaningless values on speech. Must be **excluded** for both domains. |
| **RMS Energy** | Recording-dependent (**EXCLUDED**) | Useful for arousal | ⚠️ Needs Change | While RMS is unreliable for music (varies with mastering), it is **more useful** for emotion detection where loudness correlates with arousal (angry = loud, sad = quiet). Should be **RE-INCLUDED** for the emotion task. |
| **Feature Normalization** | MinMaxScaler (per dataset) | MinMaxScaler (per dataset) | ✅ Yes | Same scaling approach works, but the scaler must be **re-fitted** on emotion data — cannot reuse music scaler parameters directly, as feature distributions differ. |
| **Data Augmentation** | Not used (10,000 samples) | May be needed | ⚠️ Needs Change | RAVDESS has fewer samples (~1,440 vs ~10,000 for GTZAN). Time-stretching, pitch-shifting, or noise injection may be needed to prevent overfitting. |

---

## Summary

### What Transfers Directly (No Changes Needed)
1. **MFCC extraction** — Universal audio feature, works for both domains
2. **Spectral Centroid & ZCR** — Capture energy and noisiness patterns applicable to both
3. **Spectral Bandwidth & Rolloff** — Compatible spectral features
4. **Feature normalization** — Same approach, re-fit on new data

### What Needs Modification
1. **Segment duration** — May need shorter windows (1-2s) for short emotion utterances
2. **RMS Energy** — Should be re-included for emotion (useful for arousal detection)
3. **Chroma features** — Less valuable for emotion; consider replacing with pitch contour
4. **Data augmentation** — Required due to smaller emotion datasets

### What Does NOT Transfer
1. **Tempo** — Meaningless for speech, must remain excluded
2. **Music-specific scaler parameters** — Feature distributions differ, need re-fitting
