"""
Persona effects figure: structural-choice rate by model and institutional persona.
Descriptive rates (choice = B) straight from the database, so all seven models are
shown, including Qwen (whose per-model regression did not converge). Complements the
model x question heatmap (Fig. 1) and the variance decomposition (Fig. 2) by showing
the contextually activated bias of Section 4.4.
"""
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt

plt.style.use(["science", "nature", "no-latex"])

DB = "data/experiment.db"
PERSONAS = ["neutral", "ipcc", "unep", "worldbank"]
PERSONA_LABEL = {"neutral": "Neutral", "ipcc": "IPCC",
                 "unep": "UNEP", "worldbank": "World Bank"}
PERSONA_COLOR = {"neutral": "#7f7f7f", "ipcc": "#d6604d",
                 "unep": "#1a9850", "worldbank": "#4575b4"}
NAME = {"gpt-4o": "GPT-4o", "claude-sonnet-4-5": "Claude Sonnet 4.5",
        "mistral-large": "Mistral Large", "deepseek-v3": "DeepSeek V3",
        "qwen2.5-72b": "Qwen 2.5-72B", "llama-3.1-70b-local": "Llama 3.1-70B",
        "sabia-3": "Sabiá-3"}

con = sqlite3.connect(DB)
rows = con.execute(
    """select model_name, persona,
              100.0*sum(case when choice='B' then 1 else 0 end)/count(*)
       from runs where choice in ('A','B') group by model_name, persona"""
).fetchall()
overall = dict(con.execute(
    """select model_name,
              100.0*sum(case when choice='B' then 1 else 0 end)/count(*)
       from runs where choice in ('A','B') group by model_name"""
).fetchall())
con.close()

rate = {}
for m, p, v in rows:
    rate.setdefault(m, {})[p] = v

models = sorted(overall, key=lambda m: overall[m])  # ascending, strongest on top

fig, ax = plt.subplots(figsize=(7.0, 3.9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.axvline(50, color="#cccccc", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(50, len(models) - 0.35, "chance", color="#999999", fontsize=7.5,
        ha="center", va="bottom")

for y, m in enumerate(models):
    vals = [rate[m][p] for p in PERSONAS]
    ax.plot([min(vals), max(vals)], [y, y], color="#dddddd", lw=2.2, zorder=2)
    for p in PERSONAS:
        ax.scatter(rate[m][p], y, s=46, color=PERSONA_COLOR[p],
                   edgecolor="white", lw=0.6, zorder=3)

ax.set_yticks(range(len(models)))
ax.set_yticklabels([NAME[m] for m in models], fontsize=9)
ax.set_xlabel("Structural-choice rate (%)", fontsize=9.5)
ax.set_xlim(45, 100)
ax.set_ylim(-0.6, len(models) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)

handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=7,
                      markerfacecolor=PERSONA_COLOR[p], markeredgecolor="white",
                      label=PERSONA_LABEL[p]) for p in PERSONAS]
ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
          framealpha=0.95, edgecolor="#cccccc", title="Persona", title_fontsize=8)

ax.set_title("Institutional persona shifts the structural-choice rate, heterogeneously",
             fontsize=10.5, pad=8, color="#1a1a1a")

fig.tight_layout()
out = Path("results/figures/persona_effects.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)
