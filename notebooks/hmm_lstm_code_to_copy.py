# ==============================================================================
# ADD THESE IMPORTS AND CLASSES TO THE "Model Definitions" CELL
# ==============================================================================

# Note: Ensure you run `!pip install hmmlearn` in the first cell if you haven't.
from hmmlearn import hmm

def build_lstm(input_shape, num_classes, name='LSTM'):
    """Pure LSTM Model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name=name)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class HMMClassifier:
    """Multi-class classifier using one HMM per class."""
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.hmms = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(list(set(y)))
        for cls in self.classes:
            X_cls = X[y == cls]
            n_comp = min(self.n_components, len(X_cls) // 2)
            n_comp = max(n_comp, 1)

            model = hmm.GaussianHMM(
                n_components=n_comp, 
                covariance_type="diag", 
                n_iter=100,
                random_state=42
            )
            # For 1D features, we treat each sample as length 1
            lengths = [1] * len(X_cls)
            model.fit(X_cls, lengths)
            self.hmms[cls] = model

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            sample = X[i:i+1]
            best_score = -np.inf
            best_cls = self.classes[0]
            for cls in self.classes:
                try:
                    score = self.hmms[cls].score(sample)
                    if score > best_score:
                        best_score = score
                        best_cls = cls
                except:
                    pass
            predictions.append(best_cls)
        return np.array(predictions)


# ==============================================================================
# ADD THIS TO A NEW CELL IN "Task 1: Music Genre Classification"
# ==============================================================================

# --- Genre LSTM ---
print("="*60)
print("TRAINING LSTM FOR GENRE CLASSIFICATION")
print("="*60)

genre_lstm = build_lstm((X_train_g_cnn.shape[1], 1), len(genre_encoder.classes_), 'Genre_LSTM')
t0 = time.time()
h_glstm = genre_lstm.fit(X_train_g_cnn, y_train_g,
                         validation_data=(X_test_g_cnn, y_test_g),
                         epochs=100, batch_size=32,
                         callbacks=get_callbacks(), verbose=1)
glstm_time = time.time() - t0

y_pred_glstm = np.argmax(genre_lstm.predict(X_test_g_cnn), axis=1)
glstm_acc = accuracy_score(y_test_g, y_pred_glstm)
print(f"\n🎯 Genre LSTM Accuracy: {glstm_acc*100:.2f}%")
plot_history(h_glstm, 'Genre LSTM')
plot_cm(y_test_g, y_pred_glstm, genre_encoder.classes_, 'Genre LSTM Confusion Matrix')

# --- Genre HMM ---
print("\n" + "="*60)
print("TRAINING HMM FOR GENRE CLASSIFICATION")
print("="*60)

t0 = time.time()
genre_hmm = HMMClassifier(n_components=4)
genre_hmm.fit(X_train_g, y_train_g)
ghmm_time = time.time() - t0

y_pred_ghmm = genre_hmm.predict(X_test_g)
ghmm_acc = accuracy_score(y_test_g, y_pred_ghmm)
print(f"🎯 Genre HMM Accuracy: {ghmm_acc*100:.2f}%")
plot_cm(y_test_g, y_pred_ghmm, genre_encoder.classes_, 'Genre HMM Confusion Matrix')

# Add to summary (Run this immediately after training)
genre_results['LSTM'] = {'accuracy': glstm_acc, 'time': glstm_time}
genre_results['HMM'] = {'accuracy': ghmm_acc, 'time': ghmm_time}


# ==============================================================================
# ADD THIS TO A NEW CELL IN "Task 2: Music Emotion Detection"
# ==============================================================================

# --- Emotion LSTM ---
print("="*60)
print("TRAINING LSTM FOR EMOTION DETECTION")
print("="*60)

emotion_lstm = build_lstm((X_train_e_cnn.shape[1], 1), n_emo, 'Emotion_LSTM')
t0 = time.time()
h_elstm = emotion_lstm.fit(X_train_e_cnn, y_train_e,
                           validation_data=(X_test_e_cnn, y_test_e),
                           epochs=100, batch_size=32,
                           class_weight=emotion_class_weights,
                           callbacks=get_callbacks(), verbose=1)
elstm_time = time.time() - t0

y_pred_elstm = np.argmax(emotion_lstm.predict(X_test_e_cnn), axis=1)
elstm_acc = accuracy_score(y_test_e, y_pred_elstm)
print(f"\n🎯 Emotion LSTM Accuracy: {elstm_acc*100:.2f}%")
plot_history(h_elstm, 'Emotion LSTM')
plot_cm(y_test_e, y_pred_elstm, emotion_encoder.classes_, 'Emotion LSTM Confusion Matrix')

# --- Emotion HMM ---
print("\n" + "="*60)
print("TRAINING HMM FOR EMOTION DETECTION")
print("="*60)

t0 = time.time()
emotion_hmm = HMMClassifier(n_components=4)
emotion_hmm.fit(X_train_e, y_train_e)
ehmm_time = time.time() - t0

y_pred_ehmm = emotion_hmm.predict(X_test_e)
ehmm_acc = accuracy_score(y_test_e, y_pred_ehmm)
print(f"🎯 Emotion HMM Accuracy: {ehmm_acc*100:.2f}%")
plot_cm(y_test_e, y_pred_ehmm, emotion_encoder.classes_, 'Emotion HMM Confusion Matrix')

# Add to summary (Run this immediately after training)
emotion_results['LSTM'] = {'accuracy': elstm_acc, 'time': elstm_time}
emotion_results['HMM'] = {'accuracy': ehmm_acc, 'time': ehmm_time}


# ==============================================================================
# ADD THIS TO THE "Save All Models" CELL
# ==============================================================================

# Save Keras models
genre_lstm.save('saved_models/genre_lstm_model.keras')
emotion_lstm.save('saved_models/emotion_lstm_model.keras')

# Save HMM models
with open('saved_models/genre_hmm_model.pkl', 'wb') as f:
    pickle.dump(genre_hmm, f)
with open('saved_models/emotion_hmm_model.pkl', 'wb') as f:
    pickle.dump(emotion_hmm, f)
