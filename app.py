"""
app.py — Medical Disease Prediction System
Entry point for Streamlit Community Cloud.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="MedAI — Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# MedAI\nMedical Disease Prediction System powered by Machine Learning.",
    },
)

from utils.init_app import init_session
from utils.logger import logger

# ── initialise model (trains automatically if pkl files are missing) ───────────
init_session()

logger.info("Home page loaded.")

metrics    = st.session_state["metrics"]
le         = st.session_state["le"]
all_syms   = st.session_state["all_symptoms"]
best       = metrics.get("best_model", "Random Forest")
acc        = metrics.get(best, {}).get("accuracy", 0) * 100

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <span style='font-size:2.8rem'>🏥</span>
      <div style='font-family:sans-serif;font-size:1.3rem;font-weight:700;
                  color:#0EA5E9;margin-top:.3rem'>MedAI</div>
      <div style='font-size:.78rem;color:#94A3B8'>Disease Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🧭 Navigation")
    st.markdown("""
    - 🩺 **Disease Prediction** — predict from symptoms
    - 📚 **Disease Information** — browse disease details
    - 📊 **Model Analytics** — charts & metrics
    - ℹ️ **About** — project info
    """)
    st.divider()

    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    col1.metric("Diseases", len(le.classes_))
    col2.metric("Symptoms", len(all_syms))
    st.metric("Best Model Accuracy", f"{acc:.1f}%")
    st.caption(f"Best model: **{best}**")
    st.divider()

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <strong>Disclaimer:</strong> For educational purposes only.
    Always consult a qualified healthcare professional.
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
Select a page from the **sidebar** (tap ☰ on mobile) to begin:

| Page | Description |
|------|-------------|
| 🩺 Disease Prediction | Enter symptoms and receive an instant AI diagnosis |
| 📚 Disease Information | Browse detailed disease cards with prevention tips |
| 📊 Model Analytics | Explore model performance charts and feature importances |
| ℹ️ About | Learn about the project, dataset, and technology stack |
""")

c1, c2, c3, c4 = st.columns(4)
for col, emoji, title, desc in [
    (c1, "🧬", f"{len(le.classes_)} Diseases", "Wide range of conditions"),
    (c2, "🔬", f"{len(all_syms)} Symptoms",  "Comprehensive vocabulary"),
    (c3, "🤖", "2 ML Models",                "Random Forest + XGBoost"),
    (c4, "⚡", "Real-time",                  "Instant prediction"),
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
  Built with ❤️ using <strong>Streamlit</strong>, <strong>scikit-learn</strong>
  &amp; <strong>XGBoost</strong> &nbsp;·&nbsp; <strong>MedAI</strong> v1.0
  &nbsp;·&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)
