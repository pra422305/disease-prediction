import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix

# ──────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────
PRIMARY   = "#0EA5E9"
SECONDARY = "#6366F1"
ACCENT    = "#10B981"
WARN      = "#F59E0B"
DANGER    = "#EF4444"
BG_DARK   = "#0F172A"
BG_MID    = "#1E293B"
TEXT_LIGHT = "#F1F5F9"

PALETTE = [PRIMARY, SECONDARY, ACCENT, WARN, DANGER,
           "#A78BFA", "#34D399", "#FB923C", "#F472B6", "#38BDF8"]


def _dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_MID)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_LIGHT)
    ax.xaxis.label.set_color(TEXT_LIGHT)
    ax.yaxis.label.set_color(TEXT_LIGHT)
    ax.title.set_color(TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    return fig, ax


# ──────────────────────────────────────────────
# Matplotlib / Seaborn charts
# ──────────────────────────────────────────────

def plot_accuracy_comparison(metrics: dict):
    """Bar chart comparing RF and XGB accuracy."""
    fig, ax = _dark_fig(figsize=(8, 4))
    models = list(metrics.keys())
    accs   = [metrics[m].get("accuracy", 0) * 100 for m in models]
    bars = ax.bar(models, accs, color=[PRIMARY, SECONDARY], width=0.5, zorder=3)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.7, zorder=0)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", color=TEXT_LIGHT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, all_symptoms: list, top_n: int = 20):
    """Horizontal bar chart of top-N feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[-top_n:][::-1]
    top_syms    = [all_symptoms[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = _dark_fig(figsize=(10, 6))
    colors  = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top_syms)))
    ax.barh(range(len(top_syms)), top_vals[::-1], color=colors[::-1], zorder=3)
    ax.set_yticks(range(len(top_syms)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in top_syms[::-1]], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.7, zorder=0)
    fig.tight_layout()
    return fig


def plot_disease_distribution(df: pd.DataFrame):
    """Horizontal bar chart of disease sample distribution."""
    counts = df["Disease"].value_counts()
    fig, ax = _dark_fig(figsize=(10, 8))
    colors  = plt.cm.cool(np.linspace(0.2, 0.9, len(counts)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors, zorder=3)
    ax.set_xlabel("Number of Records", fontsize=11)
    ax.set_title("Disease Distribution in Dataset", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.7, zorder=0)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(metrics: dict):
    """Seaborn heatmap of RF vs XGB metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    data = {m: [metrics[m].get(k, 0) for k in metric_keys] for m in metrics}
    df_heat = pd.DataFrame(data, index=[k.replace("_", " ").title() for k in metric_keys])

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG_MID)
    ax.set_facecolor(BG_MID)
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="#334155", ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Model Metrics Heatmap", color=TEXT_LIGHT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT_LIGHT)
    ax.xaxis.label.set_color(TEXT_LIGHT)
    ax.yaxis.label.set_color(TEXT_LIGHT)
    plt.setp(ax.get_xticklabels(), color=TEXT_LIGHT)
    plt.setp(ax.get_yticklabels(), color=TEXT_LIGHT, rotation=0)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_chart(y_true, y_pred, labels: list):
    """Seaborn confusion matrix (subset of first 15 classes for readability)."""
    # y_true/y_pred may be integer-encoded; use integer labels for confusion_matrix
    unique_vals = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_vals)
    n       = min(15, len(unique_vals))
    cm_sub  = cm[:n, :n]
    # Map integer indices to string disease names if labels list provided
    if labels and isinstance(unique_vals[0], (int, np.integer)):
        lbl_sub = [labels[i][:12] for i in unique_vals[:n]]
    else:
        lbl_sub = [str(l)[:12] for l in unique_vals[:n]]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_MID)
    ax.set_facecolor(BG_MID)
    sns.heatmap(cm_sub, xticklabels=lbl_sub, yticklabels=lbl_sub,
                annot=True, fmt="d", cmap="Blues", linewidths=0.4,
                linecolor="#334155", ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Confusion Matrix (first 15 diseases)", color=TEXT_LIGHT,
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", color=TEXT_LIGHT)
    ax.set_ylabel("Actual",    color=TEXT_LIGHT)
    plt.setp(ax.get_xticklabels(), color=TEXT_LIGHT, rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), color=TEXT_LIGHT, rotation=0, fontsize=7)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Plotly charts
# ──────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1E293B",
    plot_bgcolor="#0F172A",
    font=dict(color="#F1F5F9", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)


def plotly_accuracy_bar(metrics: dict):
    models = list(metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    fig = go.Figure()
    colors = [PRIMARY, SECONDARY, ACCENT, WARN]
    for i, key in enumerate(metric_keys):
        vals = [metrics[m].get(key, 0) * 100 for m in models]
        fig.add_trace(go.Bar(
            name=key.replace("_", " ").title(),
            x=models, y=vals,
            marker_color=colors[i],
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Interactive Model Metrics Comparison",
        barmode="group",
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        height=420,
    )
    return fig


def plotly_confidence_chart(top5: list):
    """Horizontal bar chart for top-5 prediction probabilities."""
    diseases = [r["disease"] for r in top5]
    probs    = [r["probability"] for r in top5]
    fig = go.Figure(go.Bar(
        x=probs,
        y=diseases,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale=[[0, SECONDARY], [0.5, PRIMARY], [1, ACCENT]],
            showscale=True,
            colorbar=dict(title="% Conf.", tickfont=dict(color=TEXT_LIGHT)),
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Top-5 Prediction Probabilities",
        xaxis_title="Confidence (%)",
        height=350,
    )
    return fig


def plotly_disease_pie(df: pd.DataFrame):
    counts = df["Disease"].value_counts().reset_index()
    counts.columns = ["Disease", "Count"]
    fig = px.pie(
        counts, names="Disease", values="Count",
        color_discrete_sequence=PALETTE,
        hole=0.38,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      pull=[0.03] * len(counts))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Disease Frequency Distribution",
        showlegend=False,
        height=500,
    )
    return fig


def plotly_gauge(confidence: float, disease: str):
    color = ACCENT if confidence >= 70 else WARN if confidence >= 40 else DANGER
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        number={"suffix": "%", "font": {"color": TEXT_LIGHT, "size": 32}},
        title={"text": f"Confidence for<br><b>{disease}</b>",
               "font": {"color": TEXT_LIGHT, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_LIGHT},
            "bar": {"color": color},
            "bgcolor": BG_MID,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40],  "color": "#1E293B"},
                {"range": [40, 70], "color": "#292524"},
                {"range": [70, 100],"color": "#14532d"},
            ],
            "threshold": {
                "line": {"color": ACCENT, "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
        delta={"reference": 50, "increasing": {"color": ACCENT},
               "decreasing": {"color": DANGER}},
    ))
    fig.update_layout(
        paper_bgcolor=BG_MID,
        font=dict(color=TEXT_LIGHT),
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
