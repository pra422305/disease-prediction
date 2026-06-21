"""
pages/3_Model_Analytics.py
"""

import os, sys
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Model Analytics", page_icon="📊", layout="wide")

from utils.init_app import init_session
from utils.visualizations import (
    plot_accuracy_comparison, plot_feature_importance, plot_disease_distribution,
    plot_metrics_heatmap, plot_confusion_matrix_chart,
    plotly_accuracy_bar, plotly_confidence_chart, plotly_disease_pie,
)

init_session()

metrics  = st.session_state["metrics"]
model    = st.session_state["model"]
symptoms = st.session_state["all_symptoms"]
df       = st.session_state["df"]
cm_data  = st.session_state.get("cm_data")

display_metrics = {k: v for k, v in metrics.items() if k != "best_model"}
best_model      = metrics.get("best_model", "Random Forest")
best_acc        = display_metrics.get(best_model, {}).get("accuracy", 0) * 100

st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>📊 Model Analytics</div>
  <div class='hero-sub'>Performance metrics, feature importance, and visual insights</div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
rf = display_metrics.get("Random Forest", {})
xg = display_metrics.get("XGBoost", {})
k1.metric("🏆 Best Model",      best_model)
k2.metric("✅ Accuracy",         f"{best_acc:.1f}%")
k3.metric("🎯 F1 Score",         f"{rf.get('f1',0)*100:.1f}%")
k4.metric("🔬 Diseases Covered", str(len(df["Disease"].unique())))

st.divider()

# ── Plotly interactive charts ─────────────────────────────────────────────────
st.markdown("<div class='section-heading'>⚡ Interactive Charts (Plotly)</div>", unsafe_allow_html=True)
pc1, pc2 = st.columns(2)
with pc1:
    st.plotly_chart(plotly_accuracy_bar(display_metrics), use_container_width=True)
with pc2:
    st.plotly_chart(plotly_disease_pie(df), use_container_width=True)

# ── Matplotlib charts ─────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>📈 Model Performance (Matplotlib)</div>", unsafe_allow_html=True)
mc1, mc2 = st.columns(2)
with mc1:
    st.pyplot(plot_accuracy_comparison(display_metrics))
with mc2:
    st.pyplot(plot_feature_importance(model, symptoms))

st.pyplot(plot_disease_distribution(df))

# ── Seaborn charts ────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🔥 Seaborn Visualisations</div>", unsafe_allow_html=True)
sc1, sc2 = st.columns(2)
with sc1:
    st.pyplot(plot_metrics_heatmap(display_metrics))
with sc2:
    if cm_data:
        st.pyplot(plot_confusion_matrix_chart(
            np.array(cm_data["y_test"]),
            np.array(cm_data["y_pred"]),
            cm_data["labels"],
        ))
    else:
        st.info("Confusion matrix data not available.")

# ── Detailed metrics table ────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>📋 Detailed Metrics</div>", unsafe_allow_html=True)
import pandas as pd
rows = []
for mname, mvals in display_metrics.items():
    rows.append({
        "Model":     mname,
        "Accuracy":  f"{mvals.get('accuracy',0)*100:.2f}%",
        "Precision": f"{mvals.get('precision',0)*100:.2f}%",
        "Recall":    f"{mvals.get('recall',0)*100:.2f}%",
        "F1 Score":  f"{mvals.get('f1',0)*100:.2f}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("<div class='footer'>MedAI 🏥 · Model analytics dashboard</div>", unsafe_allow_html=True)
