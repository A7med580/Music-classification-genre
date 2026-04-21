"""
CNN (1D Convolutional Neural Network) model for Music Genre Classification.

Uses pre-extracted audio features from the GTZAN dataset CSV to classify
music into 10 genres. The 1D-CNN treats the 50 features as a 1D signal
and applies convolutional filters to learn local feature patterns.

Architecture:
    Input(n_features, 1) → Conv1D(64) → BN → Conv1D(128) → BN → Pool
    → Conv1D(256) → BN → GlobalAvgPool → Dense(256) → Dropout → Dense(10)

Typical accuracy: 90%+ on GTZAN 3-second segments.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessing import (
    load_dataset, prepare_data_split, prepare_cnn_data, ensure_output_dir
)


def build_cnn_model(input_shape, num_classes):
    """Build a 1D-CNN model for genre classification.

    Architecture uses Conv1D layers with batch normalization and dropout
    for regularization. Global average pooling reduces spatial dimensions
    before the classification head.

    Args:
        input_shape: Tuple of (n_features, 1) for Conv1D input.
        num_classes: Number of output classes (10 for GTZAN).

    Returns:
        Compiled Keras Sequential model.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Block 1
        layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Classification head
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, output_dir):
    """Plot and save training/validation accuracy and loss curves.

    Args:
        history: Keras History object from model.fit().
        output_dir: Directory to save the plot PNG files.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('CNN Training Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('CNN Training Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_training_curves.png'), dpi=150)
    plt.close()
    print(f"  Training curves saved to {output_dir}/cnn_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name='CNN'):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: True labels (integer-encoded).
        y_pred: Predicted labels (integer-encoded).
        class_names: List of class name strings.
        output_dir: Directory to save the plot.
        model_name: Name for the plot title and filename.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()

    filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {output_dir}/{filename}")


def train_cnn(epochs=50, batch_size=32):
    """Train the CNN model on GTZAN features and save results.

    Loads the dataset, prepares train/test split, builds and trains the
    1D-CNN model, then saves the model weights, training curves, confusion
    matrix, and classification report.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Dictionary with training results including accuracy and metrics.
    """
    from tensorflow import keras

    print("=" * 60)
    print("CNN MODEL TRAINING")
    print("=" * 60)

    output_dir = ensure_output_dir()

    # Load and prepare data
    print("\n[1/5] Loading dataset...")
    X, y, feature_names = load_dataset(exclude_features=True)
    data = prepare_data_split(X, y)
    print(f"  Features: {X.shape[1]} | Train: {data['X_train'].shape[0]} | Test: {data['X_test'].shape[0]}")

    # Reshape for Conv1D
    X_train_cnn, X_test_cnn = prepare_cnn_data(data['X_train'], data['X_test'])
    print(f"  CNN input shape: {X_train_cnn.shape[1:]}")

    # Build model
    print("\n[2/5] Building CNN model...")
    model = build_cnn_model(
        input_shape=(X_train_cnn.shape[1], 1),
        num_classes=data['num_classes']
    )
    model.summary()

    # Train
    print(f"\n[3/5] Training for {epochs} epochs...")
    start_time = time.time()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train_cnn, data['y_train_encoded'],
        validation_data=(X_test_cnn, data['y_test_encoded']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s")

    # Evaluate
    print("\n[4/5] Evaluating...")
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(data['y_test_encoded'], y_pred)

    report = classification_report(
        data['y_test_encoded'], y_pred,
        target_names=data['class_names'], output_dict=True
    )
    report_str = classification_report(
        data['y_test_encoded'], y_pred,
        target_names=data['class_names']
    )

    print(f"\n  CNN Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n{report_str}")

    # Save results
    print("\n[5/5] Saving model and plots...")
    model.save(os.path.join(output_dir, 'cnn_model.keras'))
    plot_training_history(history, output_dir)
    plot_confusion_matrix(
        data['y_test_encoded'], y_pred,
        data['class_names'], output_dir, 'CNN'
    )

    # Save metrics
    results = {
        'model': 'CNN (1D-Conv)',
        'accuracy': float(accuracy),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_macro': float(report['macro avg']['f1-score']),
        'training_time_seconds': round(training_time, 1),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'per_genre': {
            name: {
                'precision': float(report[name]['precision']),
                'recall': float(report[name]['recall']),
                'f1': float(report[name]['f1-score']),
            }
            for name in data['class_names']
        }
    }

    with open(os.path.join(output_dir, 'cnn_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"CNN TRAINING COMPLETE — Accuracy: {accuracy*100:.1f}%")
    print(f"{'=' * 60}\n")

    return results


if __name__ == '__main__':
    train_cnn()
