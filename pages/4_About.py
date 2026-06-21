"""
pages/4_About.py
"""

import os
import sys
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

css_path = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
  <div class='hero-title'>ℹ️ About MedAI</div>
  <div class='hero-sub'>An open-source educational disease prediction system built with ML & Streamlit</div>
</div>
""", unsafe_allow_html=True)

# ── Project description ───────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🏥 Project Overview</div>", unsafe_allow_html=True)
st.markdown("""
**MedAI — Medical Disease Prediction System** is a production-ready, end-to-end machine-learning
application that predicts potential diseases based on a user-selected set of symptoms.

The system is trained on a curated symptom–disease dataset covering **41 diseases** and
**132+ symptoms**, using two powerful ensemble classifiers:
- **Random Forest Classifier** (scikit-learn)
- **XGBoost Classifier** (XGBoost)

The best-performing model is automatically selected and saved for inference. The entire
application is deployable with a single command on **Streamlit Community Cloud**.
""")

# ── Features ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>✨ Key Features</div>", unsafe_allow_html=True)
fc1, fc2 = st.columns(2)
features_left = [
    ("🩺", "Symptom-based Disease Prediction", "Select up to 5+ symptoms and receive an instant AI diagnosis with confidence score."),
    ("📊", "Model Analytics Dashboard",         "Compare Random Forest vs XGBoost with accuracy, precision, recall, and F1 metrics."),
    ("📚", "Disease Information Library",        "Search 41 diseases with detailed descriptions, causes, symptoms, and prevention tips."),
    ("📈", "Interactive Visualisations",         "Plotly, Matplotlib, and Seaborn charts for comprehensive data insights."),
]
features_right = [
    ("🎯", "Confidence Gauge",          "Real-time confidence score displayed as an interactive Plotly gauge."),
    ("🔒", "Input Validation",          "Robust error handling and validation prevents invalid predictions."),
    ("📝", "Comprehensive Logging",     "All events, predictions, and errors are logged to logs/app.log."),
    ("☁️", "Cloud-Ready Deployment",    "Optimised for Streamlit Community Cloud and GitHub deployment."),
]
for col, feats in [(fc1, features_left), (fc2, features_right)]:
    for icon, title, desc in feats:
        col.markdown(f"""
        <div class='info-card'>
          <h4>{icon} {title}</h4>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
tc1, tc2, tc3, tc4 = st.columns(4)
stacks = [
    ("🖥️ Frontend", ["Streamlit", "Custom CSS", "Plotly", "Matplotlib", "Seaborn"]),
    ("🤖 Machine Learning", ["scikit-learn", "XGBoost", "Pandas", "NumPy", "Pickle / Joblib"]),
    ("📦 Backend", ["Python 3.9+", "Logging", "JSON", "Pathlib", "OS / Sys"]),
    ("☁️ Deployment", ["Streamlit Community Cloud", "GitHub", "requirements.txt"]),
]
for col, (title, items) in zip([tc1, tc2, tc3, tc4], stacks):
    items_html = "".join(f"<li>{i}</li>" for i in items)
    col.markdown(f"""
    <div class='metric-card' style='text-align:left'>
      <div style='font-family:"Space Grotesk",sans-serif;font-weight:600;
                  color:#0EA5E9;margin-bottom:.6rem'>{title}</div>
      <ul style='color:#94A3B8;font-size:.88rem;padding-left:1.2rem;margin:0'>
        {items_html}
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ── ML workflow ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>🔄 Machine Learning Workflow</div>",
            unsafe_allow_html=True)
steps = [
    ("1", "Data Loading",      "Load disease_dataset.csv with Disease + Symptom columns"),
    ("2", "Preprocessing",     "Clean data, strip whitespace, normalise symptom names"),
    ("3", "Feature Engineering","One-hot encode symptoms into a binary feature matrix"),
    ("4", "Label Encoding",    "Encode disease names to integers with LabelEncoder"),
    ("5", "Train/Test Split",  "80/20 stratified split to preserve class distribution"),
    ("6", "Model Training",    "Train Random Forest & XGBoost classifiers"),
    ("7", "Evaluation",        "Compute Accuracy, Precision, Recall, F1 Score"),
    ("8", "Model Selection",   "Compare F1 scores — best model is automatically saved"),
    ("9", "Persistence",       "Save model, encoder, symptoms list, and metrics as pickle/JSON"),
]
wc1, wc2, wc3 = st.columns(3)
for i, (num, title, desc) in enumerate(steps):
    col = [wc1, wc2, wc3][i % 3]
    col.markdown(f"""
    <div class='info-card' style='margin-bottom:.6rem'>
      <div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.3rem'>
        <span style='background:linear-gradient(135deg,#0EA5E9,#6366F1);
                     color:#fff;border-radius:50%;width:24px;height:24px;
                     display:flex;align-items:center;justify-content:center;
                     font-size:.75rem;font-weight:700;flex-shrink:0'>{num}</span>
        <span style='color:#F1F5F9;font-weight:600;font-size:.92rem'>{title}</span>
      </div>
      <p style='margin:0'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Dataset info ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>📂 Dataset Information</div>", unsafe_allow_html=True)
st.markdown("""
| Property | Value |
|----------|-------|
| File | `data/disease_dataset.csv` |
| Format | CSV — 1 Disease column + 5 Symptom columns |
| Total Diseases | 41 |
| Total Symptoms | 132+ |
| Total Records | ~160 rows (4 variants per disease) |
| Source | Based on the Kaggle Disease Symptom Prediction dataset |
""")

# ── Folder structure ──────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>📁 Project Structure</div>", unsafe_allow_html=True)
st.code("""
Medical-Disease-Prediction/
├── app.py                          ← Main entry point
├── requirements.txt
├── README.md
├── model/
│   ├── train_model.py              ← Training script
│   ├── disease_prediction_model.pkl
│   ├── label_encoder.pkl
│   ├── symptoms_list.pkl
│   ├── model_metrics.json
│   └── cm_data.pkl
├── data/
│   └── disease_dataset.csv
├── utils/
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── logger.py
│   └── visualizations.py
├── pages/
│   ├── 1_Disease_Prediction.py
│   ├── 2_Disease_Information.py
│   ├── 3_Model_Analytics.py
│   └── 4_About.py
├── assets/
│   └── styles.css
├── disease_info/
│   └── disease_details.json
└── logs/
    └── app.log
""", language="text")

# ── Developer info ────────────────────────────────────────────────────────────
st.markdown("<div class='section-heading'>👨‍💻 Developer</div>", unsafe_allow_html=True)
st.markdown("""
<div class='metric-card' style='max-width:500px;text-align:left'>
  <div style='font-size:2.5rem;margin-bottom:.5rem'>👨‍💻</div>
  <div style='font-family:"Space Grotesk",sans-serif;font-size:1.2rem;
              font-weight:700;color:#F1F5F9'>MedAI Development Team</div>
  <div style='color:#94A3B8;font-size:.9rem;margin-top:.3rem'>
    Full Stack ML Engineer · Python · Streamlit · scikit-learn
  </div>
  <div style='margin-top:1rem;color:#94A3B8;font-size:.88rem'>
    Built as a production-ready demonstration of an end-to-end
    machine learning application for academic and hackathon submissions.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class='warning-box'>
⚠️ <strong>Important Disclaimer:</strong> MedAI is developed purely for <strong>educational and
research purposes</strong>. The predictions generated by this system are based on a simplified
machine-learning model and should <strong>never</strong> be used as a substitute for professional
medical advice, diagnosis, or treatment. Always consult a qualified and licensed healthcare
professional for any medical concerns.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  MedAI 🏥 v1.0 &nbsp;·&nbsp; Built with ❤️ using Streamlit, scikit-learn & XGBoost
  &nbsp;·&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)
