"""
pages/2_Disease_Information.py
"""

import os
import sys
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Disease Information", page_icon="📚", layout="wide")

css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "disease_info" not in st.session_state:
    st.error("⚠️ Please navigate to the **Home** page first to initialise the app.")
    st.stop()

disease_info = st.session_state["disease_info"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>📚 Disease Information</div>
  <div class='hero-sub'>Comprehensive database covering 41 diseases — descriptions, causes, prevention & more</div>
</div>
""", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🔍 Search Disease</div>", unsafe_allow_html=True)
search_col, filter_col = st.columns([3, 1])
with search_col:
    query = st.text_input("Search by disease name or keyword", placeholder="e.g. Diabetes, fever, liver…")
with filter_col:
    severity_filter = st.selectbox("Filter by Severity", ["All", "Low", "Medium", "High", "Critical"])

# ── Filter logic ──────────────────────────────────────────────────────────────
def matches(name: str, info: dict) -> bool:
    if severity_filter != "All" and info.get("severity", "") != severity_filter:
        return False
    if query:
        q = query.lower()
        text = (name + " " + info.get("description", "") +
                " " + info.get("symptoms", "")).lower()
        return q in text
    return True

filtered = {k: v for k, v in disease_info.items() if matches(k, v)}

# ── Stats row ─────────────────────────────────────────────────────────────────
sev_counts = {}
for info in disease_info.values():
    s = info.get("severity", "Unknown")
    sev_counts[s] = sev_counts.get(s, 0) + 1

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Total Diseases", len(disease_info))
mc2.metric("🟢 Low",      sev_counts.get("Low", 0))
mc3.metric("🟡 Medium",   sev_counts.get("Medium", 0))
mc4.metric("🔴 High",     sev_counts.get("High", 0))
mc5.metric("💀 Critical", sev_counts.get("Critical", 0))

st.markdown(f"**Showing {len(filtered)} disease(s)**")
st.divider()

# ── Disease cards ─────────────────────────────────────────────────────────────
severity_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "💀"}
severity_class = {
    "Low": "badge-low", "Medium": "badge-medium",
    "High": "badge-high", "Critical": "badge-critical"
}

if not filtered:
    st.markdown("""
    <div class='warning-box'>
    🔍 No diseases found for your search. Try a different keyword or clear the filter.
    </div>
    """, unsafe_allow_html=True)
else:
    for disease, info in filtered.items():
        sev     = info.get("severity", "Unknown")
        sev_em  = severity_emoji.get(sev, "⚪")
        sev_cls = severity_class.get(sev, "badge-medium")
        doctor  = info.get("doctor", "General Physician")

        with st.expander(f"{sev_em} {disease}  —  👨‍⚕️ {doctor}", expanded=False):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"""
                <div class='info-card'>
                  <h4>📖 Description</h4>
                  <p>{info.get('description', '—')}</p>
                </div>
                <div class='info-card'>
                  <h4>🦠 Causes</h4>
                  <p>{info.get('causes', '—')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class='info-card'>
                  <h4>🩺 Common Symptoms</h4>
                  <p>{info.get('symptoms', '—')}</p>
                </div>
                <div class='info-card'>
                  <h4>🛡️ Prevention</h4>
                  <p>{info.get('prevention', '—')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:flex;gap:1rem;margin-top:.5rem;align-items:center'>
              <span>Severity: <span class='badge {sev_cls}'>{sev}</span></span>
              <span style='color:#94A3B8'>|</span>
              <span style='color:#0EA5E9'>👨‍⚕️ Recommended Specialist: <strong>{doctor}</strong></span>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>MedAI 🏥 · Disease Information Library · For educational use only</div>
""", unsafe_allow_html=True)
