import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/features_3_sec.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

print("Loading dataset...")
data = pd.read_csv(DATA_PATH)

# Drop filename and length if they exist (based on CSV header)
# filename, length, ... features ... label
X = data.drop(['filename', 'length', 'label'], axis=1)
y = data['label']

print("Training model...")
# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split (optional, but good for verification if we wanted it)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.4f}")

# Save artifacts
print("Saving model and scaler...")
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Save label list for decoding if needed (though model.classes_ usually works, explicit is safer)
joblib.dump(model.classes_, ENCODER_PATH)

print("Done.")
