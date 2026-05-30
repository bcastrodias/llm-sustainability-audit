"""
Generates a publication-quality heatmap: questions (rows) x models (columns)
showing % structural choices. Uses seaborn for clean cell rendering.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import binomtest

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

DB_PATH = Path("data/experiment.db")
OUT_PATH = Path("results/figures/results_heatmap.png")

SHORT = {
    "claude-sonnet-4-5":   "Claude",
    "gpt-4o":              "GPT-4o",
    "mistral-large":       "Mistral",
    "llama-3.1-70b-local": "Llama",
    "deepseek-v3":         "DeepSeek",
    "qwen2.5-72b":         "Qwen",
    "sabia-3":             "Sabia",
}

MODEL_ORDER = ["Claude", "Mistral", "Llama", "DeepSeek", "Qwen", "Sabia", "GPT-4o"]

Q_LABELS = {
    "Q01": "Q01  Transition mechanism",
    "Q02": "Q02  Externalities",
    "Q03": "Q03  Ecosystem services",
    "Q04": "Q04  Transition scope",
    "Q05": "Q05  Policy neutrality (irr.)",
    "Q06": "Q06  Speed of action",
    "Q07": "Q07  Jevons paradox",
    "Q08": "Q08  Central bank",
    "Q09": "Q09  Instrument mix",
    "Q10": "Q10  Global governance",
    "Q11": "Q11  Climate debt",
    "Q12": "Q12  Knowledge systems (irr.)",
    "Q13": "Q13  Growth paradigm",
    "Q14": "Q14  Nature valuation",
}

CONTESTED   = {"Q07", "Q08", "Q10", "Q13"}
IRREDUTIBLE = {"Q05", "Q12"}

# ── load data ──────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT choice, model_name, question_id FROM runs WHERE choice IN ('A','B')",
    conn
)
conn.close()

df["model"] = df["model_name"].map(SHORT).fillna(df["model_name"])
df["structural"] = (df["choice"] == "B").astype(int)

pivot = (
    df.groupby(["question_id", "model"])["structural"]
    .mean().mul(100).round(0)
    .unstack("model")
    .reindex(columns=MODEL_ORDER)
    .reindex(list(Q_LABELS.keys()))
)

def _binom_p(series):
    k = int(series.sum()); n = int(series.count())
    return binomtest(k, n, 0.5).pvalue if n > 0 else np.nan

pval_pivot = (
    df.groupby(["question_id", "model"])["structural"]
    .apply(_binom_p)
    .unstack("model")
    .reindex(columns=MODEL_ORDER)
    .reindex(list(Q_LABELS.keys()))
)

# ── annotation matrix: "87%***" on one line ────────────────────────────────
annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
for qid in pivot.index:
    for col in pivot.columns:
        val = pivot.loc[qid, col]
        p   = pval_pivot.loc[qid, col]
        if pd.isna(val):
            annot.loc[qid, col] = "—"
        else:
            annot.loc[qid, col] = f"{int(val)}%{stars(p)}"

# ── colormap ───────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "bias", ["#2166ac", "#f7f7f7", "#d6604d", "#b2182b"], N=256
)
norm = mcolors.Normalize(vmin=0, vmax=100)

# ── figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6.0))

sns.heatmap(
    pivot,
    annot=annot,
    fmt="",
    cmap=cmap,
    norm=norm,
    ax=ax,
    linewidths=0.4,
    linecolor="#dddddd",
    annot_kws={"size": 7, "va": "center"},
    cbar_kws={"shrink": 0.5, "label": "% structural choices",
              "orientation": "horizontal", "pad": 0.12},
)

# ── row background tints (irredutible / contested) ─────────────────────────
q_list = list(Q_LABELS.keys())
for i, qid in enumerate(q_list):
    if qid in IRREDUTIBLE:
        ax.add_patch(plt.Rectangle(
            (-0.02, len(q_list) - i - 1), len(MODEL_ORDER) + 0.02, 1,
            transform=ax.transData, color="#ffe8e8", zorder=0, clip_on=False
        ))
    elif qid in CONTESTED:
        ax.add_patch(plt.Rectangle(
            (-0.02, len(q_list) - i - 1), len(MODEL_ORDER) + 0.02, 1,
            transform=ax.transData, color="#e8eeff", zorder=0, clip_on=False
        ))

# ── fix annotation text colour (white on dark cells) ──────────────────────
for text in ax.texts:
    try:
        row_txt, col_txt = text.get_position()
        # seaborn places text at cell centres in data coords
        col_idx = int(col_txt)
        row_idx = int(row_txt)
        val = pivot.iloc[row_idx, col_idx]
        if pd.isna(val):
            text.set_color("#aaaaaa")
            continue
        rgba = cmap(norm(val))
        luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        text.set_color("white" if luminance < 0.45 else "#1a1a1a")
        if val >= 99 or val <= 10:
            text.set_fontweight("bold")
    except Exception:
        pass

# ── axes labels ────────────────────────────────────────────────────────────
ax.set_yticklabels(
    [Q_LABELS[q] for q in reversed(q_list)],
    fontsize=7.5, rotation=0
)
ax.set_xticklabels(MODEL_ORDER, fontsize=8, rotation=0)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(left=False, bottom=False)

# ── fix colorbar font ──────────────────────────────────────────────────────
cbar = ax.collections[0].colorbar
cbar.set_label("% structural choices", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks([0, 25, 50, 75, 100])

# ── legend ─────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor="#ffe8e8", edgecolor="#ccc",
                   label="Irredutible: 100% in all models/conditions"),
    mpatches.Patch(facecolor="#e8eeff", edgecolor="#ccc",
                   label="Contested: high cross-model variance"),
]
ax.legend(handles=patches, loc="upper left",
          bbox_to_anchor=(0.0, -0.18), ncol=1,
          fontsize=7, frameon=False)

fig.text(0.5, -0.04,
         "Binomial test vs 50%:  *** p<0.001  ** p<0.01  * p<0.05  ns not significant",
         ha="center", fontsize=7, color="#555555")

plt.title("% Structural choices by question and model\n(all personas, languages, temperatures)",
          fontsize=9, pad=10)

fig.tight_layout()
OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
