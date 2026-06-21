"""
utils/init_app.py
Shared initializer — call init_session() at the top of every page.
Ensures model is trained and loaded regardless of which page the user lands on.
"""

import os
import sys
import pickle
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_model():
    """Auto-train if pkl artefacts are missing (cold Streamlit Cloud start)."""
    required = [
        os.path.join(ROOT, "model", "disease_prediction_model.pkl"),
        os.path.join(ROOT, "model", "label_encoder.pkl"),
        os.path.join(ROOT, "model", "symptoms_list.pkl"),
        os.path.join(ROOT, "model", "model_metrics.json"),
    ]
    if not all(os.path.exists(p) for p in required):
        from model.train_model import train
        train()


@st.cache_resource(show_spinner="🔄 Loading AI model — please wait…")
def _load_resources():
    _ensure_model()
    from utils.predictor import load_model, load_metrics, load_disease_info
    import pandas as pd

    model, le, symptoms = load_model()
    metrics = load_metrics()
    disease_info = load_disease_info()
    df = pd.read_csv(os.path.join(ROOT, "data", "disease_dataset.csv"))

    cm_data = None
    try:
        with open(os.path.join(ROOT, "model", "cm_data.pkl"), "rb") as f:
            cm_data = pickle.load(f)
    except Exception:
        pass

    return model, le, symptoms, metrics, disease_info, df, cm_data


def _inject_css():
    css_path = os.path.join(ROOT, "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session():
    """
    Call once at the top of every page file.
    Loads model into st.session_state if not already present.
    Returns True when ready, never shows the error-and-stop message.
    """
    _inject_css()

    if "model" not in st.session_state:
        (
            st.session_state["model"],
            st.session_state["le"],
            st.session_state["all_symptoms"],
            st.session_state["metrics"],
            st.session_state["disease_info"],
            st.session_state["df"],
            st.session_state["cm_data"],
        ) = _load_resources()

    return True
