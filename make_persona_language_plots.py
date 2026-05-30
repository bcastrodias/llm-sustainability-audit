"""
Generates two figures:
  1. Forest plot — persona effects (OR vs neutral, with 95% CI)
  2. Horizontal bar chart — language effect (PT vs EN OR) per model
"""
import matplotlib
matplotlib.use("Agg")
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

plt.style.use(["science", "nature", "no-latex"])

conn = sqlite3.connect("data/experiment.db")
df = pd.read_sql(
    "SELECT choice, model_name, persona, language FROM runs WHERE choice IN ('A','B')",
    conn
)
conn.close()
df["structural"] = (df["choice"] == "B").astype(int)

SHORT = {
    "claude-sonnet-4-5":   "Claude",
    "gpt-4o":              "GPT-4o",
    "mistral-large":       "Mistral",
    "llama-3.1-70b-local": "Llama",
    "deepseek-v3":         "DeepSeek",
    "qwen2.5-72b":         "Qwen",
    "sabia-3":             "Sabiá",
}
df["model"] = df["model_name"].map(SHORT).fillna(df["model_name"])

def crosstab_or(data, group_col, a_val, b_val, outcome_col):
    sub = data[data[group_col].isin([a_val, b_val])]
    t = pd.crosstab(sub[group_col], sub[outcome_col])
    for col in [0, 1]:
        if col not in t.columns:
            t[col] = 0
    OR = (t.loc[a_val, 1] * t.loc[b_val, 0]) / (t.loc[a_val, 0] * t.loc[b_val, 1])
    se = np.sqrt(sum(1 / max(v, 0.5) for v in t.values.flatten()))
    lo = np.exp(np.log(OR) - 1.96 * se)
    hi = np.exp(np.log(OR) + 1.96 * se)
    return OR, lo, hi

# ── Figure 1: Persona forest plot ─────────────────────────────────────────────
personas = [
    ("unep",      "UNEP",        "#e07b39"),
    ("ipcc",      "IPCC",        "#5a8fc2"),
    ("worldbank", "World Bank",  "#4caf7d"),
]

fig1, ax1 = plt.subplots(figsize=(3.5, 2.4))   # Nature single-column

ys = [2, 1, 0]
for (code, label, color), y in zip(personas, ys):
    OR, lo, hi = crosstab_or(df, "persona", code, "neutral", "structural")
    ax1.plot([lo, hi], [y, y], color=color, lw=1.5, solid_capstyle="round", zorder=2)
    ax1.scatter(OR, y, color=color, s=40, zorder=3)
    ax1.text(hi + 0.03, y, f"OR = {OR:.2f}  [{lo:.2f}–{hi:.2f}]",
             va="center", fontsize=6, color="#1a1a1a")

ax1.axvline(1.0, color="#888888", lw=0.8, ls="--", zorder=1)
ax1.set_yticks(ys)
ax1.set_yticklabels([p[1] for p in personas], fontsize=7)
ax1.set_xlabel("Odds Ratio (vs. neutral persona)  [95% CI]", fontsize=7)
ax1.set_xlim(0.35, 2.3)
ax1.set_ylim(-0.6, 2.6)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.tick_params(left=False, labelsize=7)
ax1.set_title("Persona framing effects on structural choices", fontsize=8, pad=6)

ax1.annotate("more structural", xy=(1.7, -0.52), fontsize=6,
             color="#888", ha="center",
             arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.6),
             xytext=(2.1, -0.52))
ax1.annotate("less structural", xy=(0.7, -0.52), fontsize=6,
             color="#888", ha="center",
             arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.6),
             xytext=(0.38, -0.52))

fig1.tight_layout()
out1 = Path("results/figures/persona_forest_plot.png")
out1.parent.mkdir(exist_ok=True)
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")

# ── Figure 2: Language OR per model (PT vs EN) ────────────────────────────────
MODEL_ORDER = ["GPT-4o", "Mistral", "Llama", "DeepSeek", "Claude", "Qwen", "Sabiá"]

lang_data = []
for model in MODEL_ORDER:
    mdf = df[df["model"] == model]
    OR, lo, hi = crosstab_or(mdf, "language", "pt", "en", "structural")
    lang_data.append((model, OR, lo, hi))

fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))   # Nature single-column

ys2 = list(range(len(MODEL_ORDER)))
for (model, OR, lo, hi), y in zip(lang_data, ys2):
    color = "#d6604d" if OR > 1.0 else "#2166ac"
    ax2.barh(y, OR - 1.0, left=1.0, color=color, alpha=0.75, height=0.5, zorder=2)
    ax2.plot([lo, hi], [y, y], color="#333333", lw=1.0,
             solid_capstyle="round", zorder=3)
    ax2.scatter(OR, y, color="#333333", s=20, zorder=4)
    xpos = hi + 0.03 if OR >= 1 else lo - 0.03
    ha = "left" if OR >= 1 else "right"
    ax2.text(xpos, y, f"{OR:.2f}", va="center", fontsize=6,
             ha=ha, color="#1a1a1a")

ax2.axvline(1.0, color="#555555", lw=0.8, ls="--", zorder=1)
ax2.set_yticks(ys2)
ax2.set_yticklabels(MODEL_ORDER, fontsize=7)
ax2.set_xlabel("Odds Ratio  PT vs EN  [95% CI]", fontsize=7)
ax2.set_xlim(0.45, 1.85)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.tick_params(left=False, labelsize=7)
ax2.set_title("Effect of Portuguese (vs. English) on structural choices, by model",
              fontsize=8, pad=6)

patches = [
    mpatches.Patch(facecolor="#d6604d", alpha=0.75, label="PT increases structural bias"),
    mpatches.Patch(facecolor="#2166ac", alpha=0.75, label="PT decreases structural bias"),
]
ax2.legend(handles=patches, fontsize=6, frameon=False,
           loc="lower right", bbox_to_anchor=(1.0, 0.0))

fig2.tight_layout()
out2 = Path("results/figures/language_or_by_model.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")
