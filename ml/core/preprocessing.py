"""
Preprocessing module for Music Genre Classification.

Handles loading the GTZAN dataset from CSV, splitting data for training,
and preparing inputs for CNN, GMM, and Random Forest models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Data')
DATA_PATH_3SEC = os.path.join(DATA_DIR, 'features_3_sec.csv')
DATA_PATH_30SEC = os.path.join(DATA_DIR, 'features_30_sec.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Features to EXCLUDE based on analysis
FEATURES_TO_EXCLUDE = [
    'tempo',           # Highly variable within genres
    'harmony_mean',    # Redundant with MFCCs
    'harmony_var',     # Redundant with MFCCs
    'perceptr_mean',   # Redundant with MFCCs
    'perceptr_var',    # Redundant with MFCCs
    'rms_mean',        # Recording-level dependent
    'rms_var',         # Recording-level dependent
]

# All genre labels in the GTZAN dataset
GENRE_LABELS = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]


def load_dataset(csv_path=None, exclude_features=False):
    """Load the GTZAN dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file. Defaults to features_3_sec.csv.
        exclude_features: If True, removes features identified as unhelpful
            (tempo, harmony, perceptr, RMS) from the feature set.

    Returns:
        Tuple of (X, y, feature_names) where:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of genre labels (strings)
            feature_names: list of feature column names
    """
    if csv_path is None:
        csv_path = DATA_PATH_3SEC

    data = pd.read_csv(csv_path)

    # Drop non-feature columns
    drop_cols = ['filename', 'length', 'label']

    if exclude_features:
        drop_cols.extend([f for f in FEATURES_TO_EXCLUDE if f in data.columns])

    X = data.drop([c for c in drop_cols if c in data.columns], axis=1)
    feature_names = list(X.columns)
    y = data['label'].values

    return X.values, y, feature_names


def load_raw_dataframe(csv_path=None):
    """Load the raw DataFrame without any processing.

    Args:
        csv_path: Path to the CSV file. Defaults to features_3_sec.csv.

    Returns:
        pandas DataFrame with all columns.
    """
    if csv_path is None:
        csv_path = DATA_PATH_3SEC
    return pd.read_csv(csv_path)


def get_label_encoder(y=None):
    """Get a fitted LabelEncoder for genre labels.

    Args:
        y: Optional array of labels to fit on. If None, uses GENRE_LABELS.

    Returns:
        Fitted sklearn LabelEncoder.
    """
    le = LabelEncoder()
    if y is not None:
        le.fit(y)
    else:
        le.fit(GENRE_LABELS)
    return le


def prepare_data_split(X, y, test_size=0.2, random_state=42, scale=True):
    """Split and optionally scale the data for model training.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        scale: If True, applies MinMaxScaler to features.

    Returns:
        Dictionary with keys:
            'X_train', 'X_test': scaled feature matrices
            'y_train', 'y_test': label arrays (strings)
            'y_train_encoded', 'y_test_encoded': integer-encoded labels
            'scaler': fitted MinMaxScaler (or None)
            'label_encoder': fitted LabelEncoder
    """
    le = get_label_encoder(y)
    y_encoded = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_encoded': y_train_enc,
        'y_test_encoded': y_test_enc,
        'scaler': scaler,
        'label_encoder': le,
        'num_classes': len(le.classes_),
        'class_names': list(le.classes_),
    }


def prepare_cnn_data(X_train, X_test):
    """Reshape features for 1D-CNN input.

    The 1D-CNN expects input shape (n_samples, n_features, 1) to apply
    Conv1D filters along the feature dimension.

    Args:
        X_train: Training features of shape (n_samples, n_features).
        X_test: Test features of shape (n_samples, n_features).

    Returns:
        Tuple of (X_train_cnn, X_test_cnn) with shape (n_samples, n_features, 1).
    """
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train_cnn, X_test_cnn


def ensure_output_dir():
    """Create the outputs directory if it doesn't exist.

    Returns:
        Path to the outputs directory.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    return OUTPUTS_DIR
