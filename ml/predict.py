"""
Inference script for Music Genre Classification.

Loads the high-performance CNN model and performs genre prediction on
uploaded audio files. This script is called as a subprocess by the
Node.js backend.

Inputs:
    - Path to an audio file (e.g., .wav, .mp3) provided as command line argument.

Outputs:
    - JSON object containing 'genre' and 'confidence' to stdout.
"""

import sys
import os
import json
import numpy as np
import joblib
import warnings

# Suppress TensorFlow and audio processing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add current directory to path to import core modules
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

try:
    import tensorflow as tf
    from core.features import extract_features_from_audio
    from core.preprocessing import get_label_encoder
except ImportError as e:
    print(json.dumps({"error": f"Missing dependencies: {str(e)}. Please run setup_env.sh."}))
    sys.exit(1)

# Paths to artifacts
# We use the outputs directory which contains our best CNN model
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'outputs', 'cnn_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'outputs', 'scaler.pkl')

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"Audio file not found at: {audio_path}"}))
        sys.exit(1)

    # Check for model files
    if not os.path.exists(CNN_MODEL_PATH):
        # Fallback to old path if outputs directory not structured yet
        # or if the user is testing before full pipeline run
        print(json.dumps({"error": "CNN model not found. Please run ml/train_all_models.py first."}))
        sys.exit(1)

    try:
        # 1. Load model and artifacts
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = get_label_encoder()
        
        # 2. Extract features
        # Same logic used during training (3s duration, 0.5s offset)
        features = extract_features_from_audio(audio_path, duration=3, offset=0.5)
        
        # 3. Preprocess
        # CNN expects shape (1, n_features, 1)
        features_scaled = scaler.transform([features])
        features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
        
        # 4. Predict
        predictions = model.predict(features_cnn, verbose=0)
        class_idx = np.argmax(predictions[0])
        genre = label_encoder.inverse_transform([class_idx])[0]
        confidence = float(predictions[0][class_idx])
        
        # Output result as JSON for the Node.js backend
        result = {
            "genre": genre,
            "confidence": confidence,
            "model_type": "CNN (1D-Convolutional)"
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
