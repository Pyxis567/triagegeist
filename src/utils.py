import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 93

# ── Triage level palette (Level 1 → Level 5) ────────────────────────────────
ACUITY_COLORS = ["#D7191C", "#F46D43", "#FDAE61", "#A6D96A", "#1A9641"]

ACUITY_LABELS = {
    1: "L1\nResuscitation",
    2: "L2\nEmergent",
    3: "L3\nUrgent",
    4: "L4\nLess Urgent",
    5: "L5\nNon-Urgent",
}

PLOTLY_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#e5e5e5",
})


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, ax):
    """Seaborn heatmap with exact project style (Blues, %, linewidths=0.5)."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = [f"L{c}" for c in classes]
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "%"},
    )
    ax.set_title("Confusion Matrix (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)


def plot_feature_importance(importances, ax, top_n=20):
    """Horizontal bar chart: top-N features, RdYlGn ramp low→high, mean vline."""
    top = importances.sort_values(ascending=True).tail(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
    top.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(top.mean(), color="red", linestyle="--", linewidth=1, label="mean")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.legend()


def plot_f1_by_class(report, classes, macro_f1, ax):
    """Bar chart of per-class F1 using ACUITY_COLORS; macro F1 as dashed line."""
    f1_scores = [report[str(c)]["f1-score"] for c in classes]
    x_labels = [ACUITY_LABELS[c].replace("\n", " ") for c in classes]
    bar_colors = [ACUITY_COLORS[c - 1] for c in classes]

    ax.bar(x_labels, f1_scores, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.axhline(macro_f1, color="black", linestyle="--", linewidth=1,
               label=f"Macro-F1: {macro_f1:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 Score by Triage Level", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.xaxis.grid(False)
    ax.legend()
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    """Print accuracy + macro F1 + classification report. Return (acc, macro_f1)."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("=" * 45)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print("=" * 45)
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, macro_f1
