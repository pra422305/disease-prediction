"""
pages/3_Model_Analytics.py
"""

import os
import sys
import pickle
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Model Analytics", page_icon="📊", layout="wide")

css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "model" not in st.session_state:
    st.error("⚠️ Please navigate to the **Home** page first to initialise the app.")
    st.stop()

model        = st.session_state["model"]
all_symptoms = st.session_state["all_symptoms"]
metrics      = st.session_state["metrics"]
df           = st.session_state["df"]
cm_data      = st.session_state.get("cm_data")

from utils.visualizations import (
    plot_accuracy_comparison,
    plot_feature_importance,
    plot_disease_distribution,
    plot_metrics_heatmap,
    plot_confusion_matrix_chart,
    plotly_accuracy_bar,
    plotly_disease_pie,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>📊 Model Analytics</div>
  <div class='hero-sub'>Deep-dive into model performance, feature importances, and dataset insights</div>
</div>
""", unsafe_allow_html=True)

# ── Key metrics banner ────────────────────────────────────────────────────────
best = metrics.get("best_model", "Random Forest")
bm   = metrics.get(best, {})
metric_keys = ["accuracy", "precision", "recall", "f1_score"]
labels      = ["Accuracy", "Precision", "Recall", "F1 Score"]
cols = st.columns(len(metric_keys))
for col, key, label in zip(cols, metric_keys, labels):
    col.metric(label, f"{bm.get(key, 0)*100:.2f}%")
st.caption(f"🏆 Best model: **{best}**")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📉 Matplotlib / Seaborn", "📈 Plotly Interactive", "🗂️ Dataset Insights"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Matplotlib / Seaborn
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-heading'>Matplotlib Charts</div>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Model Accuracy Comparison**")
        display_metrics = {k: v for k, v in metrics.items() if k != "best_model"}
        if len(display_metrics) >= 1:
            fig = plot_accuracy_comparison(display_metrics)
            st.pyplot(fig)

    with r1c2:
        st.markdown("**Feature Importance (Top 20)**")
        if hasattr(model, "feature_importances_"):
            fig = plot_feature_importance(model, all_symptoms, top_n=20)
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model type.")

    st.markdown("<div class='section-heading'>Disease Distribution</div>",
                unsafe_allow_html=True)
    fig = plot_disease_distribution(df)
    st.pyplot(fig)

    st.markdown("<div class='section-heading'>Seaborn Charts</div>",
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)

    with s1:
        st.markdown("**Metrics Heatmap**")
        display_metrics = {k: v for k, v in metrics.items() if k != "best_model"}
        if len(display_metrics) >= 1:
            fig = plot_metrics_heatmap(display_metrics)
            st.pyplot(fig)

    with s2:
        st.markdown("**Confusion Matrix**")
        if cm_data:
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            le = st.session_state["le"]
            y_test = np.array(cm_data["y_test"])
            y_pred = np.array(cm_data["y_pred"])
            labels = le.classes_.tolist()
            fig = plot_confusion_matrix_chart(y_test, y_pred, labels)
            st.pyplot(fig)
        else:
            st.info("Confusion matrix data not available. Re-train the model.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Plotly Interactive
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-heading'>Interactive Model Metrics</div>",
                unsafe_allow_html=True)
    display_metrics = {k: v for k, v in metrics.items() if k != "best_model"}
    if len(display_metrics) >= 1:
        fig = plotly_accuracy_bar(display_metrics)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-heading'>Disease Frequency Pie Chart</div>",
                unsafe_allow_html=True)
    fig = plotly_disease_pie(df)
    st.plotly_chart(fig, use_container_width=True)

    # Confidence chart placeholder — show RF metrics as bar
    st.markdown("<div class='section-heading'>Per-Model F1 Score</div>",
                unsafe_allow_html=True)
    import plotly.graph_objects as go
    model_names = [k for k in metrics if k != "best_model"]
    f1_vals     = [metrics[m].get("f1_score", 0) * 100 for m in model_names]
    fig = go.Figure(go.Bar(
        x=model_names, y=f1_vals,
        marker_color=["#0EA5E9", "#6366F1"][:len(model_names)],
        text=[f"{v:.2f}%" for v in f1_vals],
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="#1E293B", plot_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        title="F1 Score per Model",
        yaxis_title="F1 Score (%)",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Dataset Insights
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-heading'>Dataset Overview</div>",
                unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Total Records",  len(df))
    dc2.metric("Unique Diseases", df["Disease"].nunique())
    dc3.metric("Unique Symptoms", len(all_symptoms))

    st.markdown("**Sample Records**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Disease Record Counts**")
    counts = df["Disease"].value_counts().reset_index()
    counts.columns = ["Disease", "Records"]
    st.dataframe(counts, use_container_width=True, height=400)

    st.markdown("**All Recognised Symptoms**")
    import pandas as pd
    sym_df = pd.DataFrame({"Symptom": [s.replace("_"," ").title() for s in all_symptoms]})
    st.dataframe(sym_df, use_container_width=True, height=300)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>MedAI 🏥 · Model Analytics Dashboard · Powered by scikit-learn & XGBoost</div>
""", unsafe_allow_html=True)
