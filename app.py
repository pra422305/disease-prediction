"""
app.py — Medical Disease Prediction System
Entry point for Streamlit Community Cloud.
"""

import os
import sys
import json
import pickle
import streamlit as st

# ── ensure project root is on path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.logger import logger

# ── auto-train if model artefacts are missing ─────────────────────────────────
def _ensure_model():
    required = [
        "model/disease_prediction_model.pkl",
        "model/label_encoder.pkl",
        "model/symptoms_list.pkl",
        "model/model_metrics.json",
    ]
    if not all(os.path.exists(p) for p in required):
        logger.info("Model artefacts not found — running train_model.py …")
        from model.train_model import train
        train()

_ensure_model()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI — Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "# MedAI — Medical Disease Prediction System\nPowered by Machine Learning.",
    },
)

# ── inject CSS ────────────────────────────────────────────────────────────────
css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── load shared resources into session_state ──────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_resources():
    from utils.predictor import load_model, load_metrics, load_disease_info
    model, le, symptoms = load_model()
    metrics = load_metrics()
    disease_info = load_disease_info()
    import pandas as pd
    df = pd.read_csv("data/disease_dataset.csv")
    # load confusion matrix data if available
    cm_data = None
    try:
        with open("model/cm_data.pkl", "rb") as f:
            cm_data = pickle.load(f)
    except Exception:
        pass
    return model, le, symptoms, metrics, disease_info, df, cm_data

(
    st.session_state["model"],
    st.session_state["le"],
    st.session_state["all_symptoms"],
    st.session_state["metrics"],
    st.session_state["disease_info"],
    st.session_state["df"],
    st.session_state["cm_data"],
) = _load_resources()

logger.info("App loaded — resources cached.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <span style='font-size:2.8rem'>🏥</span>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.3rem;
                  font-weight:700;color:#0EA5E9;margin-top:.3rem'>MedAI</div>
      <div style='font-size:.78rem;color:#94A3B8'>Disease Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🧭 Navigation")
    st.markdown("""
    Use the **pages** listed below to explore:
    - 🩺 **Disease Prediction** — select symptoms and get a prediction
    - 📚 **Disease Information** — search disease details
    - 📊 **Model Analytics** — charts and model performance
    - ℹ️ **About** — project info
    """)
    st.divider()

    # Quick stats
    metrics = st.session_state["metrics"]
    best = metrics.get("best_model", "Random Forest")
    acc  = metrics.get(best, {}).get("accuracy", 0) * 100
    n_diseases = len(st.session_state["le"].classes_)
    n_symptoms = len(st.session_state["all_symptoms"])

    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    col1.metric("Diseases", n_diseases)
    col2.metric("Symptoms", n_symptoms)
    st.metric("Best Model Accuracy", f"{acc:.1f}%")
    st.caption(f"Best model: **{best}**")
    st.divider()

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
    Always consult a qualified healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>🏥 Medical Disease Prediction System</div>
  <div class='hero-sub'>
    AI-powered symptom analysis using Random Forest &amp; XGBoost
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Getting Started")
st.markdown("""
Select a page from the **sidebar** to begin:

| Page | Description |
|------|-------------|
| 🩺 Disease Prediction | Enter symptoms and receive an instant AI diagnosis |
| 📚 Disease Information | Browse detailed disease cards with prevention tips |
| 📊 Model Analytics | Explore model performance charts and feature importances |
| ℹ️ About | Learn about the project, dataset, and technology stack |
""")

# Feature cards
c1, c2, c3, c4 = st.columns(4)
for col, emoji, title, desc in [
    (c1, "🧬", "41 Diseases", "Covers a wide range of conditions"),
    (c2, "🔬", "132 Symptoms", "Comprehensive symptom vocabulary"),
    (c3, "🤖", "2 ML Models", "Random Forest + XGBoost"),
    (c4, "⚡", "Real-time", "Instant prediction with confidence score"),
]:
    col.markdown(f"""
    <div class='metric-card'>
      <div style='font-size:2rem'>{emoji}</div>
      <div class='metric-value' style='font-size:1.4rem'>{title}</div>
      <div class='metric-label'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
  Built with ❤️ using <strong>Streamlit</strong>, <strong>scikit-learn</strong> &amp; <strong>XGBoost</strong>
  | <strong>MedAI</strong> v1.0 &nbsp;·&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)
