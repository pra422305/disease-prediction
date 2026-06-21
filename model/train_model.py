"""
train_model.py
Run this once to train models and persist all artefacts:
    python model/train_model.py
"""

import os
import sys
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocessing import clean_dataset, get_all_symptoms, encode_features
from utils.logger import logger

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed — will only train Random Forest.")

# ── paths ─────────────────────────────────────
DATA_PATH    = "data/disease_dataset.csv"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "disease_prediction_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SYMPTOMS_PATH= os.path.join(MODEL_DIR, "symptoms_list.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
CM_DATA_PATH = os.path.join(MODEL_DIR, "cm_data.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred,        average="weighted", zero_division=0), 4),
        "y_pred":    y_pred.tolist(),
    }, name


def train():
    logger.info("=== Starting model training ===")

    # 1. Load + clean data
    df = pd.read_csv(DATA_PATH)
    df = clean_dataset(df)
    logger.info(f"Dataset loaded: {df.shape}")

    # 2. Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Disease"])

    # 3. Build feature matrix
    all_symptoms = get_all_symptoms(df)
    X = encode_features(df, all_symptoms)
    y = df["label"].values

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=None,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics, _ = evaluate(rf, X_test, y_test, "Random Forest")
    logger.info(f"Random Forest → accuracy={rf_metrics['accuracy']}")

    # 6. XGBoost (optional)
    metrics = {"Random Forest": {k: v for k, v in rf_metrics.items() if k != "y_pred"}}
    best_model = rf
    best_name  = "Random Forest"
    y_pred_best = rf_metrics["y_pred"]

    if XGBOOST_AVAILABLE:
        n_classes = len(le.classes_)
        xgb = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_metrics, _ = evaluate(xgb, X_test, y_test, "XGBoost")
        logger.info(f"XGBoost → accuracy={xgb_metrics['accuracy']}")
        metrics["XGBoost"] = {k: v for k, v in xgb_metrics.items() if k != "y_pred"}

        if xgb_metrics["f1_score"] > rf_metrics["f1_score"]:
            best_model = xgb
            best_name  = "XGBoost"
            y_pred_best = xgb_metrics["y_pred"]
            logger.info("XGBoost selected as best model.")
        else:
            logger.info("Random Forest selected as best model.")

    metrics["best_model"] = best_name

    # 7. Save artefacts
    with open(MODEL_PATH, "wb")   as f: pickle.dump(best_model, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)
    with open(SYMPTOMS_PATH, "wb")as f: pickle.dump(all_symptoms, f)
    with open(METRICS_PATH, "w")  as f: json.dump(metrics, f, indent=2)

    # Save confusion matrix data
    cm_data = {"y_test": y_test.tolist(), "y_pred": y_pred_best,
               "labels": le.classes_.tolist()}
    with open(CM_DATA_PATH, "wb") as f: pickle.dump(cm_data, f)

    logger.info(f"All artefacts saved to {MODEL_DIR}/")
    logger.info(f"Best model: {best_name} — Accuracy: {metrics[best_name]['accuracy']*100:.2f}%")
    print("\n✅ Training complete!")
    print(f"   Best model : {best_name}")
    print(f"   Accuracy   : {metrics[best_name]['accuracy']*100:.2f}%")
    print(f"   F1 Score   : {metrics[best_name]['f1_score']*100:.2f}%")
    return best_model, le, all_symptoms, metrics


if __name__ == "__main__":
    train()
