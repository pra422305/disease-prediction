"""
pages/2_Disease_Information.py
"""

import os, sys
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Disease Information", page_icon="📚", layout="wide")

from utils.init_app import init_session

init_session()

disease_info = st.session_state.get("disease_info", {})

SEVERITY_COLOR = {
    "Low":      ("#D1FAE5", "#065F46"),
    "Medium":   ("#FEF3C7", "#92400E"),
    "High":     ("#FEE2E2", "#991B1B"),
    "Critical": ("#F3E8FF", "#6B21A8"),
}

st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>📚 Disease Information</div>
  <div class='hero-sub'>Search and explore detailed information about diseases</div>
</div>
""", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🔍 Search Disease</div>", unsafe_allow_html=True)
col_s, col_f = st.columns([3, 1])
with col_s:
    search_query = st.text_input("", placeholder="Type disease name e.g. Diabetes, Malaria…", label_visibility="collapsed")
with col_f:
    sev_filter = st.selectbox("Severity", ["All", "Low", "Medium", "High", "Critical"], label_visibility="collapsed")

all_diseases = sorted(disease_info.keys())
if search_query:
    all_diseases = [d for d in all_diseases if search_query.lower() in d.lower()]
if sev_filter != "All":
    all_diseases = [d for d in all_diseases if disease_info[d].get("severity") == sev_filter]

st.markdown(f"**{len(all_diseases)} disease(s) found**")

if not all_diseases:
    st.info("No diseases match your search. Try a different term.")
else:
    # Quick-select
    selected = st.selectbox("Select a disease to view full details:", ["— Choose —"] + all_diseases)

    if selected != "— Choose —":
        info = disease_info[selected]
        sev  = info.get("severity", "Medium")
        bg, fg = SEVERITY_COLOR.get(sev, ("#E0F2FE", "#0369A1"))

        st.markdown(f"""
        <div style='background:{bg};border-radius:16px;padding:1.5rem 2rem;margin:1rem 0'>
          <h2 style='color:{fg};margin:0'>{selected}</h2>
          <span style='background:{fg};color:white;padding:.2rem .8rem;border-radius:999px;
                font-size:.8rem;font-weight:600'>{sev} Severity</span>
          &nbsp;
          <span style='color:{fg};font-weight:600'>👨‍⚕️ {info.get("doctor","General Physician")}</span>
        </div>
        """, unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"""
            <div class='info-card'><h4>📖 Description</h4><p>{info.get('description','—')}</p></div>
            <div class='info-card'><h4>🦠 Causes</h4><p>{info.get('causes','—')}</p></div>
            """, unsafe_allow_html=True)
        with r1c2:
            st.markdown(f"""
            <div class='info-card'><h4>🩺 Symptoms</h4><p>{info.get('symptoms','—')}</p></div>
            <div class='info-card'><h4>🛡️ Prevention</h4><p>{info.get('prevention','—')}</p></div>
            """, unsafe_allow_html=True)

    # ── All disease cards grid ─────────────────────────────────────────────
    st.markdown("<div class='section-heading'>📋 All Diseases</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, disease in enumerate(all_diseases):
        info = disease_info[disease]
        sev  = info.get("severity", "Medium")
        bg, fg = SEVERITY_COLOR.get(sev, ("#E0F2FE", "#0369A1"))
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left;margin-bottom:.8rem'>
              <div style='font-weight:700;font-size:1rem;color:#0EA5E9'>{disease}</div>
              <span style='background:{fg};color:white;padding:.15rem .6rem;border-radius:999px;
                    font-size:.72rem'>{sev}</span>
              <span style='font-size:.75rem;color:#64748B;margin-left:.5rem'>
                👨‍⚕️ {info.get("doctor","GP")}</span>
              <p style='font-size:.8rem;color:#94A3B8;margin-top:.5rem;
                        display:-webkit-box;-webkit-line-clamp:2;
                        -webkit-box-orient:vertical;overflow:hidden'>
                {info.get("description","")[:120]}…
              </p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div class='footer'>MedAI 🏥 · Disease information is for educational purposes only</div>", unsafe_allow_html=True)
