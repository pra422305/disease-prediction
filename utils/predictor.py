import pickle
import json
import numpy as np
from pathlib import Path
from utils.logger import logger
from utils.preprocessing import symptoms_to_vector


MODEL_PATH = Path("model/disease_prediction_model.pkl")
ENCODER_PATH = Path("model/label_encoder.pkl")
SYMPTOMS_PATH = Path("model/symptoms_list.pkl")
METRICS_PATH = Path("model/model_metrics.json")
DISEASE_INFO_PATH = Path("disease_info/disease_details.json")


def load_model():
    """Load the trained model, label encoder, and symptoms list."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        with open(SYMPTOMS_PATH, "rb") as f:
            all_symptoms = pickle.load(f)
        logger.info("Model, encoder, and symptom list loaded successfully.")
        return model, le, all_symptoms
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return None, None, None


def load_metrics() -> dict:
    """Load model evaluation metrics from JSON."""
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load metrics: {e}")
        return {}


def load_disease_info() -> dict:
    """Load disease information JSON."""
    try:
        with open(DISEASE_INFO_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load disease info: {e}")
        return {}


def predict_disease(symptoms: list, model, le, all_symptoms: list) -> dict:
    """
    Predict disease from a list of symptom strings.
    Returns dict with disease name, confidence, and probabilities per class.
    """
    if not symptoms:
        logger.warning("Prediction called with no symptoms.")
        return {"error": "No symptoms provided."}

    vec = symptoms_to_vector(symptoms, all_symptoms)
    proba = model.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))
    pred_disease = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx]) * 100

    # Top-5 predictions
    top5_idx = np.argsort(proba)[::-1][:5]
    top5 = [
        {"disease": le.inverse_transform([i])[0], "probability": round(float(proba[i]) * 100, 2)}
        for i in top5_idx
    ]

    logger.info(f"Prediction: {pred_disease} ({confidence:.1f}%) for symptoms: {symptoms}")
    return {
        "disease": pred_disease,
        "confidence": round(confidence, 2),
        "top5": top5,
    }
