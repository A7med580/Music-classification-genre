import sys
import os
import json
import numpy as np
import librosa
import pandas as pd
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # Features matching features_3_sec.csv structure
        # length (ignored in training but in extraction) -> 66149 for 3s
        
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        
        # RMS
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_var = np.var(cent)
        
        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(bw)
        bw_var = np.var(bw)
        
        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # Harmony and Perceptr
        harmony, perceptr = librosa.effects.hpss(y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 0.10+ returns float or array, ensure it is float
        if isinstance(tempo, np.ndarray):
             tempo = tempo[0] if len(tempo) > 0 else 0

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_feat = []
        for e in mfcc:
            mfcc_feat.append(np.mean(e))
            mfcc_feat.append(np.var(e))
            
        features = [
            chroma_stft_mean, chroma_stft_var,
            rms_mean, rms_var,
            cent_mean, cent_var,
            bw_mean, bw_var,
            rolloff_mean, rolloff_var,
            zcr_mean, zcr_var,
            harmony_mean, harmony_var,
            perceptr_mean, perceptr_var,
            tempo
        ] + mfcc_feat
        
        return features
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": "Model not found. Please run training script."}))
        sys.exit(1)

    # Load artifacts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Extract features
    features = extract_features(file_path)
    
    # Scale
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)
    confidence = np.max(probs)
    
    result = {
        "genre": prediction[0],
        "confidence": float(confidence)
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
