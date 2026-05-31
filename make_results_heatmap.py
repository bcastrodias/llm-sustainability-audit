"""
Publication-quality heatmap: questions (rows) x models (columns), % structural choices.
Generates the neutral-persona version (paper Figure 1) and the all-personas version.
Colour bar on the right; the row-tint legend and significance note sit below, no overlap.
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

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

DB_PATH = Path("data/experiment.db")
SHORT = {
    "claude-sonnet-4-5": "Claude", "gpt-4o": "GPT-4o", "mistral-large": "Mistral",
    "llama-3.1-70b-local": "Llama", "deepseek-v3": "DeepSeek", "qwen2.5-72b": "Qwen",
    "sabia-3": "Sabiá",
}
MODEL_ORDER = ["Claude", "Mistral", "Llama", "DeepSeek", "Qwen", "Sabiá", "GPT-4o"]
Q_LABELS = {
    "Q01": "Q01  Transition mechanism", "Q02": "Q02  Externalities",
    "Q03": "Q03  Ecosystem services", "Q04": "Q04  Transition scope",
    "Q05": "Q05  Policy neutrality (irr.)", "Q06": "Q06  Speed of action",
    "Q07": "Q07  Jevons paradox", "Q08": "Q08  Central bank",
    "Q09": "Q09  Instrument mix", "Q10": "Q10  Global governance",
    "Q11": "Q11  Climate debt", "Q12": "Q12  Knowledge systems (irr.)",
    "Q13": "Q13  Growth paradigm", "Q14": "Q14  Nature valuation",
}
CONTESTED = {"Q07", "Q08", "Q10", "Q13"}
IRREDUCIBLE = {"Q05", "Q12"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def make_heatmap(persona, out_path, subtitle):
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT choice, model_name, question_id FROM runs WHERE choice IN ('A','B')"
    if persona:
        sql += f" AND persona = '{persona}'"
    df = pd.read_sql_query(sql, conn)
    conn.close()
    df["model"] = df["model_name"].map(SHORT).fillna(df["model_name"])
    df["structural"] = (df["choice"] == "B").astype(int)

    pivot = (df.groupby(["question_id", "model"])["structural"].mean().mul(100).round(0)
             .unstack("model").reindex(columns=MODEL_ORDER).reindex(list(Q_LABELS.keys())))

    def _p(s):
        k, n = int(s.sum()), int(s.count())
        return binomtest(k, n, 0.5).pvalue if n > 0 else np.nan
    pval = (df.groupby(["question_id", "model"])["structural"].apply(_p)
            .unstack("model").reindex(columns=MODEL_ORDER).reindex(list(Q_LABELS.keys())))

    annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for qid in pivot.index:
        for col in pivot.columns:
            v, p = pivot.loc[qid, col], pval.loc[qid, col]
            annot.loc[qid, col] = "—" if pd.isna(v) else f"{int(v)}%{stars(p)}"

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bias", ["#2166ac", "#f7f7f7", "#d6604d", "#b2182b"], N=256)
    norm = mcolors.Normalize(vmin=0, vmax=100)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sns.heatmap(pivot, annot=annot, fmt="", cmap=cmap, norm=norm, ax=ax,
                linewidths=0.4, linecolor="#dddddd", annot_kws={"size": 7, "va": "center"},
                cbar_kws={"shrink": 0.6, "label": "% structural choices", "pad": 0.02})

    q_list = list(Q_LABELS.keys())
    for i, qid in enumerate(q_list):   # pivot row i (Q01..Q14) is at y=[i, i+1], Q01 at top
        tint = "#ffe8e8" if qid in IRREDUCIBLE else "#e8eeff" if qid in CONTESTED else None
        if tint:
            ax.add_patch(plt.Rectangle((-0.02, i), len(MODEL_ORDER) + 0.02, 1,
                         transform=ax.transData, color=tint, zorder=0, clip_on=False))

    for text in ax.texts:
        try:
            c, r = int(text.get_position()[0]), int(text.get_position()[1])
            v = pivot.iloc[r, c]
            if pd.isna(v):
                text.set_color("#aaaaaa"); continue
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text.set_color("white" if lum < 0.45 else "#1a1a1a")
            if v >= 99 or v <= 10:
                text.set_fontweight("bold")
        except Exception:
            pass

    ax.set_yticklabels([Q_LABELS[q] for q in q_list], fontsize=7.5, rotation=0)
    ax.set_xticklabels(MODEL_ORDER, fontsize=8, rotation=0)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(left=False, bottom=False)

    cbar = ax.collections[0].colorbar
    cbar.set_label("% structural choices", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 25, 50, 75, 100])

    patches = [
        mpatches.Patch(facecolor="#ffe8e8", edgecolor="#ccc",
                       label="Irreducible: 100% in all models and conditions"),
        mpatches.Patch(facecolor="#e8eeff", edgecolor="#ccc",
                       label="Contested: high cross-model variance"),
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=2, fontsize=7.5, frameon=False, handlelength=1.4, columnspacing=1.6)
    ax.text(0.5, -0.155, "Binomial test vs 50%:  *** p<0.001   ** p<0.01   * p<0.05   ns not significant",
            transform=ax.transAxes, ha="center", fontsize=7, color="#555555")

    ax.set_title(f"% Structural choices by question and model\n({subtitle})", fontsize=9.5, pad=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


make_heatmap("neutral", "results/figures/results_heatmap_neutral.png",
             "neutral persona, all languages and temperatures")
make_heatmap(None, "results/figures/results_heatmap.png",
             "all personas, languages and temperatures")
