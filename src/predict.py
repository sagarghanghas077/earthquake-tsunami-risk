import joblib
import numpy as np

MODEL_PATH = "models/tsunami_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_tsunami(features: list):
    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = int(probability >= 0.5)
    return prediction, probability
