"""
Variance decomposition figure — two panels, self-explanatory layout.
Left:  total variance split (question content vs. experimental factors)
Right: within-question variance by experimental factor (lollipop)
"""
import matplotlib
matplotlib.use("Agg")
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.style.use(["science", "nature", "no-latex"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),   # Nature double-column
                                gridspec_kw={"width_ratios": [1, 1.3]})
fig.patch.set_facecolor("white")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — total variance partition
# ══════════════════════════════════════════════════════════════════════════════
ax1.set_facecolor("white")

q_val   = 27.5
exp_val = 72.5
colors  = ["#e07b39", "#dce8f7"]

# single stacked horizontal bar
ax1.barh(0.5, q_val,           color=colors[0], height=0.28, zorder=2)
ax1.barh(0.5, exp_val, left=q_val, color=colors[1], height=0.28, zorder=2,
         edgecolor="#aaaaaa", lw=0.5)

# % labels inside bars
ax1.text(q_val / 2, 0.5, f"{q_val:.1f}%",
         ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=3)
ax1.text(q_val + exp_val / 2, 0.5, f"{exp_val:.1f}%",
         ha="center", va="center", fontsize=8, fontweight="bold", color="#555", zorder=3)

# callout labels above/below bar
ax1.text(q_val / 2, 0.78,
         "Question content\n(pre-training corpus)",
         ha="center", va="bottom", fontsize=6.5, color="#e07b39", fontweight="bold")
ax1.annotate("", xy=(q_val / 2, 0.65), xytext=(q_val / 2, 0.78),
             arrowprops=dict(arrowstyle="-|>", color="#e07b39", lw=0.8))

ax1.text(q_val + exp_val / 2, 0.20,
         "Experimental factors\n(model, persona, language…)",
         ha="center", va="top", fontsize=6.5, color="#5a8fc2", fontweight="bold")
ax1.annotate("", xy=(q_val + exp_val / 2, 0.36), xytext=(q_val + exp_val / 2, 0.21),
             arrowprops=dict(arrowstyle="-|>", color="#5a8fc2", lw=0.8))

# key insight box
ax1.text(50, -0.08,
         "The dominant driver of structural bias is what is asked,\nnot who asks or how it is framed.",
         ha="center", va="top", fontsize=6, color="#333",
         style="italic", wrap=True,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7f0",
                   edgecolor="#e07b39", lw=0.6))

ax1.set_xlim(0, 100)
ax1.set_ylim(-0.55, 1.05)
ax1.axis("off")
ax1.set_title("What share of total variance does\neach source explain?",
              fontsize=7.5, pad=6, color="#1a1a1a")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — within-question variance by factor (lollipop)
# ══════════════════════════════════════════════════════════════════════════════
ax2.set_facecolor("white")

factors = ["Persona\nframing", "Model\nidentity", "Language\n(PT vs EN)", "Temperature\n(0 vs 1)"]
vals    = [2.8, 2.6, 0.0, 0.0]
colors2 = ["#5a8fc2", "#d6604d", "#aaaaaa", "#aaaaaa"]
xs = np.arange(len(factors))

# light horizontal reference lines
for yref in [1, 2, 3]:
    ax2.axhline(yref, color="#dddddd", lw=0.7, zorder=1)

ax2.axhline(0, color="#999999", lw=0.8, zorder=2)

for x, val, color in zip(xs, vals, colors2):
    ax2.plot([x, x], [0, max(val, 0.05)], color=color, lw=2,
             solid_capstyle="round", zorder=3)
    ax2.scatter(x, val, color=color, s=50, zorder=4)

    if val >= 0.1:
        ax2.text(x, val + 0.18, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=7,
                 fontweight="bold", color=color)
    else:
        ax2.text(x, 0.22, "< 0.1%",
                 ha="center", va="bottom", fontsize=6,
                 color="#aaaaaa", style="italic")

ax2.annotate("Statistically\nsignificant\n& substantive",
             xy=(0.5, 2.7), xytext=(1.7, 3.4),
             fontsize=6, color="#333", ha="center",
             arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.7,
                             connectionstyle="arc3,rad=-0.2"))

ax2.annotate("",
             xy=(1.5, 2.55), xytext=(1.7, 3.35),
             arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.7,
                             connectionstyle="arc3,rad=0.2"))

ax2.set_xticks(xs)
ax2.set_xticklabels(factors, fontsize=6.5, color="#1a1a1a")
ax2.set_ylabel("% of within-question variance explained", fontsize=6.5)
ax2.set_ylim(-0.3, 4.8)
ax2.set_xlim(-0.5, len(factors) - 0.5)
ax2.spines[["top", "right", "bottom"]].set_visible(False)
ax2.tick_params(bottom=False, left=True, labelsize=6.5)
ax2.set_title("Among experimental factors, which\ndrives the most variation?",
              fontsize=7.5, pad=6, color="#1a1a1a")

fig.suptitle("What drives structural bias?  Variance decomposition",
             fontsize=8.5, y=1.02, color="#1a1a1a", fontweight="bold")

fig.tight_layout(pad=1.5)
out = Path("results/figures/variance_decomposition.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
