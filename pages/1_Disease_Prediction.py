"""
pages/1_Disease_Prediction.py
"""

import os
import sys
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.predictor import predict_disease, load_disease_info
from utils.visualizations import plotly_confidence_chart, plotly_gauge
from utils.logger import logger

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── check resources ───────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.error("⚠️ Please navigate to the **Home** page first to initialise the app.")
    st.stop()

model        = st.session_state["model"]
le           = st.session_state["le"]
all_symptoms = st.session_state["all_symptoms"]
disease_info = st.session_state.get("disease_info", {})

# ── pretty symptom labels ─────────────────────────────────────────────────────
def pretty(s: str) -> str:
    return s.replace("_", " ").title()

symptom_display = {pretty(s): s for s in all_symptoms}
display_names   = ["— Select —"] + list(symptom_display.keys())

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>🩺 Disease Prediction</div>
  <div class='hero-sub'>Select up to 5 symptoms to receive an AI-powered diagnosis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
💡 Select the symptoms you are experiencing from the dropdowns below, then press
<strong>Predict Disease</strong>. The system will analyse your symptoms using a trained
machine-learning model and return the most likely diagnosis with a confidence score.
</div>
""", unsafe_allow_html=True)

# ── Symptom selection ─────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🔍 Symptom Selection</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col4, col5       = st.columns(2)

with col1:
    s1 = st.selectbox("Symptom 1 *", display_names, key="s1")
with col2:
    s2 = st.selectbox("Symptom 2 *", display_names, key="s2")
with col3:
    s3 = st.selectbox("Symptom 3",   display_names, key="s3")
with col4:
    s4 = st.selectbox("Symptom 4",   display_names, key="s4")
with col5:
    s5 = st.selectbox("Symptom 5",   display_names, key="s5")

# ── Advanced: multi-select ────────────────────────────────────────────────────
with st.expander("➕ Add more symptoms (advanced)", expanded=False):
    extra = st.multiselect(
        "Additional symptoms",
        options=list(symptom_display.keys()),
        help="Select any number of additional symptoms.",
    )

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔬 Predict Disease", use_container_width=True)

if predict_btn:
    # Collect selected symptoms
    raw_selected = [s1, s2, s3, s4, s5] + extra
    chosen = [symptom_display[d] for d in raw_selected if d != "— Select —"]
    chosen = list(dict.fromkeys(chosen))  # deduplicate preserving order

    if len(chosen) < 2:
        st.markdown("""
        <div class='warning-box'>
        ⚠️ Please select <strong>at least 2 symptoms</strong> for a meaningful prediction.
        </div>
        """, unsafe_allow_html=True)
    else:
        logger.info(f"Prediction request — symptoms: {chosen}")
        with st.spinner("🤖 Analysing symptoms…"):
            result = predict_disease(chosen, model, le, all_symptoms)

        if "error" in result:
            st.error(result["error"])
        else:
            disease    = result["disease"]
            confidence = result["confidence"]
            top5       = result["top5"]
            info       = disease_info.get(disease, {})

            # ── Selected symptoms summary ──────────────────────────────────
            st.markdown("<div class='section-heading'>✅ Selected Symptoms</div>",
                        unsafe_allow_html=True)
            chips = " &nbsp; ".join(
                f"<span class='badge badge-medium'>{pretty(s)}</span>"
                for s in chosen
            )
            st.markdown(f"<div style='line-height:2.2'>{chips}</div>",
                        unsafe_allow_html=True)

            # ── Prediction result card ─────────────────────────────────────
            severity = info.get("severity", "Unknown")
            sev_cls  = {
                "Low": "badge-low", "Medium": "badge-medium",
                "High": "badge-high", "Critical": "badge-critical"
            }.get(severity, "badge-medium")

            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size:3rem'>🏥</div>
              <div class='result-disease'>{disease}</div>
              <div class='result-confidence'>Confidence: {confidence:.1f}%</div>
              <div style='margin-top:.75rem'>
                Severity: <span class='badge {sev_cls}'>{severity}</span>
                &nbsp;&nbsp;
                Recommended Doctor: <span style='color:#0EA5E9;font-weight:600'>
                  {info.get("doctor", "General Physician")}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + probability chart ──────────────────────────────────
            gc, pc = st.columns([1, 2])
            with gc:
                st.plotly_chart(plotly_gauge(confidence, disease),
                                use_container_width=True)
            with pc:
                st.plotly_chart(plotly_confidence_chart(top5),
                                use_container_width=True)

            # ── Disease detail cards ───────────────────────────────────────
            if info:
                st.markdown("<div class='section-heading'>📋 Disease Information</div>",
                            unsafe_allow_html=True)
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown(f"""
                    <div class='info-card'>
                      <h4>📖 Description</h4>
                      <p>{info.get('description','—')}</p>
                    </div>
                    <div class='info-card'>
                      <h4>🦠 Causes</h4>
                      <p>{info.get('causes','—')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with dc2:
                    st.markdown(f"""
                    <div class='info-card'>
                      <h4>🛡️ Prevention</h4>
                      <p>{info.get('prevention','—')}</p>
                    </div>
                    <div class='info-card'>
                      <h4>🩺 Common Symptoms</h4>
                      <p>{info.get('symptoms','—')}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Disclaimer ─────────────────────────────────────────────────
            st.markdown("""
            <div class='warning-box' style='margin-top:1.5rem'>
            ⚠️ <strong>Medical Disclaimer:</strong> This prediction is generated by a machine-learning
            model and is intended for <strong>educational purposes only</strong>. It is not a substitute
            for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare
            provider.
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>MedAI 🏥 · For educational use only · Always consult a doctor</div>
""", unsafe_allow_html=True)
