import pandas as pd
import numpy as np
from utils.logger import logger


def get_all_symptoms(df: pd.DataFrame) -> list:
    """Extract all unique symptoms from the dataset."""
    symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
    symptoms = set()
    for col in symptom_cols:
        symptoms.update(df[col].dropna().str.strip().str.lower().unique())
    return sorted(symptoms)


def encode_features(df: pd.DataFrame, all_symptoms: list) -> pd.DataFrame:
    """One-hot encode symptoms into binary feature columns."""
    symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
    feature_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)
    for col in symptom_cols:
        for idx, val in df[col].dropna().items():
            sym = val.strip().lower()
            if sym in feature_df.columns:
                feature_df.at[idx, sym] = 1
    logger.debug(f"Encoded features: {feature_df.shape[1]} symptoms × {len(df)} rows")
    return feature_df


def symptoms_to_vector(selected_symptoms: list, all_symptoms: list):
    """Convert a user symptom selection to a binary feature DataFrame (preserves feature names)."""
    vec = np.zeros(len(all_symptoms))
    for sym in selected_symptoms:
        key = sym.strip().lower()
        if key in all_symptoms:
            vec[all_symptoms.index(key)] = 1
    return pd.DataFrame([vec], columns=all_symptoms)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise the disease column."""
    df = df.copy()
    df["Disease"] = df["Disease"].str.strip()
    sym_cols = [c for c in df.columns if c.startswith("Symptom")]
    for col in sym_cols:
        df[col] = df[col].str.strip().str.lower()
    logger.info(f"Dataset cleaned: {len(df)} rows, {df['Disease'].nunique()} diseases")
    return df
