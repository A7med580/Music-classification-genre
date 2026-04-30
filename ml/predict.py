"""
Inference script for Music Genre and Emotion Classification.

Loads the selected model and performs prediction on uploaded audio files.
This script is called as a subprocess by the Node.js backend.

Inputs:
    - Path to an audio file
    - Model type (CNN, CRNN, SVM, GMM, LSTM, HMM)

Outputs:
    - JSON object containing 'genre' and 'emotion' with confidence scores.
"""

import sys
import os
import json
import numpy as np
import joblib
import warnings
import pickle
import argparse

# Suppress TensorFlow and audio processing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

try:
    import tensorflow as tf
    from core.features import extract_features_from_audio, get_feature_names
except ImportError as e:
    print(json.dumps({"error": f"Missing dependencies: {str(e)}."}))
    sys.exit(1)

# Ensure models directory exists
MODELS_PATH = os.path.join(MODEL_DIR, 'models')

def load_sklearn_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to audio file")
    parser.add_argument('--model', type=str, default='CNN', help="Model name (CNN, CRNN, SVM, GMM, LSTM, HMM)")
    args = parser.parse_args()
    
    audio_path = args.file
    model_name = args.model
    
    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"Audio file not found at: {audio_path}"}))
        sys.exit(1)

    try:
        # Debugging paths
        # print(f"DEBUG: MODELS_PATH is {MODELS_PATH}", file=sys.stderr)
        
        # Load artifacts
        scaler_path = os.path.join(MODELS_PATH, 'feature_scaler.pkl')
        if not os.path.exists(scaler_path):
            print(json.dumps({"error": "Prediction failed", "details": f"Scaler not found at {scaler_path}. Searched in {MODELS_PATH}. Contents: {os.listdir(MODELS_PATH)}"}))
            sys.exit(1)
            
        scaler = joblib.load(scaler_path)
        genre_encoder = joblib.load(os.path.join(MODELS_PATH, 'genre_label_encoder.pkl'))
        emotion_encoder = joblib.load(os.path.join(MODELS_PATH, 'emotion_label_encoder.pkl'))
        
        with open(os.path.join(MODELS_PATH, 'training_config.json'), 'r') as f:
            config = json.load(f)
            
        feature_cols = config['feature_columns']
        all_feature_names = get_feature_names()
        
        # Get indices of the 48 selected features
        indices = [all_feature_names.index(col) for col in feature_cols]
        
        # Extract 57 features
        raw_features = extract_features_from_audio(audio_path, duration=3, offset=0.5)
        
        # Filter down to 48 features
        selected_features = raw_features[indices]
        
        # Scale
        features_scaled = scaler.transform([selected_features])
        
        # Preprocess based on model type
        is_dl = model_name in ['CNN', 'CRNN', 'LSTM']
        if is_dl:
            features_input = features_scaled.reshape(1, features_scaled.shape[1], 1)
        else:
            features_input = features_scaled

        # Load specific model files
        ext = 'keras' if is_dl else 'pkl'
        g_model_path = os.path.join(MODELS_PATH, f'genre_{model_name.lower()}_model.{ext}')
        e_model_path = os.path.join(MODELS_PATH, f'emotion_{model_name.lower()}_model.{ext}')
        
        if not os.path.exists(g_model_path) or not os.path.exists(e_model_path):
            print(json.dumps({"error": f"Model files for {model_name} not found."}))
            sys.exit(1)

        if is_dl:
            try:
                # Try loading with default settings first
                g_model = tf.keras.models.load_model(g_model_path)
                e_model = tf.keras.models.load_model(e_model_path)
            except Exception as e:
                # If it fails (e.g. quantization_config error), try loading without safe mode
                # or with custom objects if needed.
                try:
                    g_model = tf.keras.models.load_model(g_model_path, safe_mode=False)
                    e_model = tf.keras.models.load_model(e_model_path, safe_mode=False)
                except Exception as e2:
                    print(json.dumps({
                        "error": "Model Loading Failed",
                        "details": f"First attempt: {str(e)}\nSecond attempt (safe_mode=False): {str(e2)}"
                    }))
                    sys.exit(1)
            
            g_preds = g_model.predict(features_input, verbose=0)[0]
            e_preds = e_model.predict(features_input, verbose=0)[0]
            
            g_idx = np.argmax(g_preds)
            e_idx = np.argmax(e_preds)
            
            g_label = genre_encoder.inverse_transform([g_idx])[0]
            e_label = emotion_encoder.inverse_transform([e_idx])[0]
            
            g_conf = float(g_preds[g_idx])
            e_conf = float(e_preds[e_idx])
            
        else:
            if model_name in ['SVM', 'GMM']:
                g_model = load_sklearn_model(g_model_path)
                e_model = load_sklearn_model(e_model_path)
            else:
                # HMM or other pickle
                g_model = load_sklearn_model(g_model_path)
                e_model = load_sklearn_model(e_model_path)

            g_label_enc = g_model.predict(features_input)[0]
            e_label_enc = e_model.predict(features_input)[0]
            
            if model_name == 'HMM':
                g_label = genre_encoder.inverse_transform([g_label_enc])[0]
                e_label = emotion_encoder.inverse_transform([e_label_enc])[0]
                g_conf = 0.0
                e_conf = 0.0
            else:
                g_label = genre_encoder.inverse_transform([g_label_enc])[0] if not isinstance(g_label_enc, str) else g_label_enc
                e_label = emotion_encoder.inverse_transform([e_label_enc])[0] if not isinstance(e_label_enc, str) else e_label_enc
                
                try:
                    g_proba = g_model.predict_proba(features_input)[0]
                    e_proba = e_model.predict_proba(features_input)[0]
                    g_conf = float(np.max(g_proba))
                    e_conf = float(np.max(e_proba))
                except:
                    g_conf = 0.0
                    e_conf = 0.0

        # Output result as JSON
        result = {
            "genre": g_label,
            "genre_confidence": g_conf,
            "emotion": e_label,
            "emotion_confidence": e_conf,
            "model_used": model_name
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "error": "Prediction failed",
            "details": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
