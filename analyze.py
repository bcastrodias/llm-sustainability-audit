"""
Full statistical analysis pipeline for:
  "Measuring Normative Preferences in Sustainability Policy:
   A Multi-Model Audit of Large Language Models"

Runs all four analyses in sequence:
  1. Binomial tests + FDR correction (model x question)
  2. Logistic regression: main effects, model x persona, model x language
  3. Logistic regression: dimension effects (no question fixed effects)
  4. Per-model binomial tests + logistic regression

Output is printed to stdout and saved to results/stats_output.txt.

Usage:
    python analyze.py
"""

import sys
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from scipy.stats import binomtest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

DB_PATH = Path("data/experiment.db")
OUT_PATH = Path("results/stats_output.txt")
OUT_PATH.parent.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Output tee: write to stdout and file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def short_name(model_name: str) -> str:
    """Readable short label for a model name."""
    mapping = {
        "claude-sonnet-4-5":    "claude",
        "deepseek-v3":          "deepseek",
        "gpt-4o":               "gpt-4o",
        "llama-3.1-70b-local":  "llama",
        "mistral-large":        "mistral",
        "qwen2.5-72b":          "qwen",
        "sabia-3":              "sabia",
    }
    return mapping.get(model_name, model_name.split("-")[0])


def print_coefs(result, skip_question=True, skip_intercept=True):
    params = result.params
    pvalues = result.pvalues
    conf = result.conf_int()
    print(f"  {'Term':<55} {'OR':>7} {'95% CI':>20} {'p':>8}")
    print("  " + "-" * 95)
    for name in params.index:
        if skip_intercept and name == "Intercept":
            continue
        if skip_question and name.startswith("C(question_id)"):
            continue
        p = pvalues[name]
        or_ = np.exp(params[name])
        lo = np.exp(conf.loc[name, 0])
        hi = np.exp(conf.loc[name, 1])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
        ci_str = f"[{lo:.2f}, {hi:.2f}]"
        label = (name
            .replace("C(model, Treatment('claude'))[T.", "model=")
            .replace("C(model, Treatment('sabia'))[T.", "model=")
            .replace("C(persona, Treatment('neutral'))[T.", "persona=")
            .replace("C(language, Treatment('en'))[T.", "language=")
            .replace("C(temp_factor, Treatment('0.0'))[T.", "temp=")
            .replace("C(dimension)[T.", "dimension=")
            .replace("]", ""))
        print(f"  {label:<55} {or_:>7.3f} {ci_str:>20} {p:>7.4f} {sig}")
    print(f"\n  Pseudo-R2 (McFadden): {result.prsquared:.4f}")
    print(f"  Log-likelihood: {result.llf:.1f}")
    print(f"  N: {int(result.nobs):,}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(conn):
    df = pd.read_sql_query("""
        SELECT r.choice, r.model_name, r.persona, r.language, r.temperature,
               r.question_id, q.dimension, q.theme
        FROM runs r
        JOIN questions q ON r.question_id = q.id
        WHERE r.choice IN ('A', 'B')
    """, conn)
    df["structural"] = (df["choice"] == "B").astype(int)
    df["temp_factor"] = df["temperature"].astype(str)
    df["model"] = df["model_name"].map(short_name)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Binomial tests + FDR
# ---------------------------------------------------------------------------

def analysis_binomial(df):
    print()
    print("=" * 80)
    print("ANALYSIS 1 — BINOMIAL TESTS: P(structural) vs 0.5, FDR-corrected (BH)")
    print("=" * 80)

    models = sorted(df["model_name"].unique())
    questions = sorted(df["question_id"].unique())

    records = []
    for m in models:
        for q in questions:
            sub = df[(df["model_name"] == m) & (df["question_id"] == q)]
            b = int(sub["structural"].sum())
            n = len(sub)
            if n == 0:
                continue
            p_raw = binomtest(b, n, p=0.5, alternative="two-sided").pvalue
            lo, hi = wilson_ci(b, n)
            records.append({
                "model": short_name(m), "model_name": m,
                "question": q, "n_structural": b, "n_incremental": n - b, "n": n,
                "pct_b": 100 * b / n, "p_raw": p_raw,
                "ci_lo": lo * 100, "ci_hi": hi * 100,
            })

    p_values = [r["p_raw"] for r in records]
    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    for i, r in enumerate(records):
        r["p_adj"] = p_adj[i]
        r["significant"] = reject[i]

    sig = [r for r in records if r["significant"]]
    not_sig = [r for r in records if not r["significant"]]
    print(f"\nTotal tests: {len(records)}  |  Significant (FDR q<0.05): {len(sig)}  |  Not significant: {len(not_sig)}")

    print(f"\n{'Model':<12} {'Q':<5} {'%B':>6} {'95% CI':>16} {'p_adj':>10}  direction")
    print("-" * 65)
    for r in sorted(sig, key=lambda x: (x["model"], x["question"])):
        ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        direction = "structural" if r["pct_b"] > 50 else "incremental"
        print(f"{r['model']:<12} {r['question']:<5} {r['pct_b']:>5.1f}% {ci:>16} {r['p_adj']:>10.4f}  {direction}")

    if not_sig:
        print(f"\nNot significant after FDR correction:")
        for r in sorted(not_sig, key=lambda x: (x["model"], x["question"])):
            ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
            print(f"  {r['model']:<12} {r['question']:<5} {r['pct_b']:>5.1f}% {ci:>16}  p_adj={r['p_adj']:.4f}")

    print()
    print("=" * 80)
    print("SUMMARY BY QUESTION: models with significant bias")
    print("=" * 80)
    print(f"\n{'Q':<5} {'sig/total':>10}  {'models with significant structural bias'}")
    print("-" * 75)
    for q in questions:
        q_recs = [r for r in records if r["question"] == q]
        q_sig = [r for r in q_recs if r["significant"]]
        sig_models_s = [r["model"] for r in q_sig if r["pct_b"] > 50]
        sig_models_i = [r["model"] for r in q_sig if r["pct_b"] <= 50]
        label = ", ".join(sig_models_s)
        if sig_models_i:
            label += f"  [increm: {', '.join(sig_models_i)}]"
        print(f"{q:<5} {len(q_sig):>4}/{len(q_recs):<6}  {label}")

    print()
    print("=" * 80)
    print("SUMMARY BY MODEL: overall bias profile")
    print("=" * 80)
    print(f"\n{'Model':<12} {'sig_structural':>15} {'sig_incremental':>16} {'not_sig':>8} {'overall_%B':>11}")
    print("-" * 65)
    for m in models:
        m_recs = [r for r in records if r["model_name"] == m]
        m_sig_s = sum(1 for r in m_recs if r["significant"] and r["pct_b"] > 50)
        m_sig_i = sum(1 for r in m_recs if r["significant"] and r["pct_b"] <= 50)
        m_not   = sum(1 for r in m_recs if not r["significant"])
        total_b = sum(r["n_structural"] for r in m_recs)
        total_n = sum(r["n"] for r in m_recs)
        overall = 100 * total_b / total_n if total_n > 0 else 0
        print(f"{short_name(m):<12} {m_sig_s:>15} {m_sig_i:>16} {m_not:>8} {overall:>10.1f}%")


# ---------------------------------------------------------------------------
# Analysis 2: Logistic regression (cross-model)
# ---------------------------------------------------------------------------

def analysis_regression(df):
    print()
    print("=" * 80)
    print("ANALYSIS 2 — LOGISTIC REGRESSION (cross-model)")
    print("  DV: structural (1=structural pole, 0=incremental pole)")
    print("  Reference: model=claude, persona=neutral, language=en, temp=0.0")
    print("=" * 80)

    print(f"\nDataset: {len(df):,} observations  |  Structural: {df['structural'].sum():,} ({100*df['structural'].mean():.1f}%)")

    ref_model = "claude" if "claude" in df["model"].values else df["model"].iloc[0]

    print()
    print("-" * 70)
    print("MODEL 1 — Main effects")
    print("-" * 70)
    m1 = smf.logit(
        f"structural ~ C(model, Treatment('{ref_model}')) "
        "+ C(persona, Treatment('neutral')) "
        "+ C(language, Treatment('en')) "
        "+ C(temp_factor, Treatment('0.0')) "
        "+ C(dimension) "
        "+ C(question_id)",
        data=df
    ).fit(disp=False)
    print_coefs(m1)

    print()
    print("-" * 70)
    print("MODEL 2 — Main effects + model x persona interaction")
    print("-" * 70)
    m2 = smf.logit(
        f"structural ~ C(model, Treatment('{ref_model}')) "
        "* C(persona, Treatment('neutral')) "
        "+ C(language, Treatment('en')) "
        "+ C(temp_factor, Treatment('0.0')) "
        "+ C(dimension) "
        "+ C(question_id)",
        data=df
    ).fit(disp=False)
    print_coefs(m2)

    print()
    print("-" * 70)
    print("MODEL 3 — Main effects + model x language interaction")
    print("-" * 70)
    m3 = smf.logit(
        f"structural ~ C(model, Treatment('{ref_model}')) "
        "+ C(persona, Treatment('neutral')) "
        f"+ C(model, Treatment('{ref_model}')):C(language, Treatment('en')) "
        "+ C(language, Treatment('en')) "
        "+ C(temp_factor, Treatment('0.0')) "
        "+ C(dimension) "
        "+ C(question_id)",
        data=df
    ).fit(disp=False)
    print_coefs(m3)

    print()
    print("-" * 70)
    print("MODEL COMPARISON (LR test vs Model 1)")
    print("-" * 70)
    for label, mfull in [("M2 (+ model x persona)", m2), ("M3 (+ model x language)", m3)]:
        lr_stat = 2 * (mfull.llf - m1.llf)
        df_diff = mfull.df_model - m1.df_model
        p = chi2.sf(lr_stat, df_diff)
        print(f"  {label}: LR={lr_stat:.1f}, df={df_diff}, p={p:.4f}")


# ---------------------------------------------------------------------------
# Analysis 3: Dimension effects
# ---------------------------------------------------------------------------

def analysis_dimensions(df):
    print()
    print("=" * 80)
    print("ANALYSIS 3 — LOGISTIC REGRESSION: dimension effects")
    print("  (no question fixed effects; dimension coefficients identified)")
    print("  Reference: Efficiency vs. Climate Justice, model=claude, persona=neutral")
    print("=" * 80)

    REF_DIM = "Efficiency vs. Climate Justice"
    ref_model = "claude" if "claude" in df["model"].values else df["model"].iloc[0]

    print(f"\n% structural by dimension:")
    print(f"  {'Dimension':<45} {'%B':>6} {'N':>7}")
    print("  " + "-" * 60)
    for dim, grp in df.groupby("dimension"):
        pct = 100 * grp["structural"].mean()
        print(f"  {dim:<45} {pct:>5.1f}% {len(grp):>7,}")

    formula = (
        f"structural ~ C(dimension, Treatment('{REF_DIM}')) "
        f"+ C(model, Treatment('{ref_model}')) "
        "+ C(persona, Treatment('neutral')) "
        "+ C(language, Treatment('en')) "
        "+ C(temp_factor, Treatment('0.0'))"
    )
    m = smf.logit(formula, data=df).fit(disp=False)

    params, pvalues, conf = m.params, m.pvalues, m.conf_int()
    print(f"\n  {'Term':<55} {'OR':>7} {'95% CI':>20} {'p':>8}")
    print("  " + "-" * 95)
    for name in params.index:
        if name == "Intercept":
            continue
        p = pvalues[name]
        or_ = np.exp(params[name])
        lo = np.exp(conf.loc[name, 0])
        hi = np.exp(conf.loc[name, 1])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
        ci_str = f"[{lo:.2f}, {hi:.2f}]"
        label = (name
            .replace(f"C(dimension, Treatment('{REF_DIM}'))[T.", "dimension=")
            .replace(f"C(model, Treatment('{ref_model}'))[T.", "model=")
            .replace("C(persona, Treatment('neutral'))[T.", "persona=")
            .replace("C(language, Treatment('en'))[T.", "language=")
            .replace("C(temp_factor, Treatment('0.0'))[T.", "temp=")
            .replace("]", ""))
        print(f"  {label:<55} {or_:>7.3f} {ci_str:>20} {p:>7.4f} {sig}")
    print(f"\n  Pseudo-R2 (McFadden): {m.prsquared:.4f}")
    print(f"  Log-likelihood: {m.llf:.1f}")
    print(f"  N: {int(m.nobs):,}")

    print(f"\n  Predicted P(structural) by dimension (ref conditions):")
    print(f"  {'Dimension':<45} {'P(struct)':>10} {'95% CI':>20}")
    print("  " + "-" * 80)
    dims = sorted(df["dimension"].unique())
    pred_rows = [{"model": ref_model, "persona": "neutral", "language": "en",
                  "temp_factor": "0.0", "dimension": d, "structural": 0} for d in dims]
    pred_df = pd.DataFrame(pred_rows)
    pred_se = m.get_prediction(pred_df).summary_frame(alpha=0.05)
    for i, d in enumerate(dims):
        p = pred_se["predicted"].iloc[i]
        lo = pred_se["ci_lower"].iloc[i]
        hi = pred_se["ci_upper"].iloc[i]
        print(f"  {d:<45} {p:>9.1%} [{lo:.1%}, {hi:.1%}]")


# ---------------------------------------------------------------------------
# Analysis 4: Per-model
# ---------------------------------------------------------------------------

def analysis_per_model(df):
    print()
    print("=" * 80)
    print("ANALYSIS 4 — PER-MODEL BINOMIAL TESTS + LOGISTIC REGRESSION")
    print("=" * 80)

    models = sorted(df["model_name"].unique())

    for m_name in models:
        mdf = df[df["model_name"] == m_name].copy()
        label = short_name(m_name)
        overall_pct = 100 * mdf["structural"].mean()

        print()
        print("=" * 70)
        print(f"MODEL: {label.upper()}  —  overall {overall_pct:.1f}% structural  (n={len(mdf):,})")
        print("=" * 70)

        # Binomial + FDR
        records = []
        for q in sorted(mdf["question_id"].unique()):
            qdf = mdf[mdf["question_id"] == q]
            b = int(qdf["structural"].sum())
            n = len(qdf)
            p_raw = binomtest(b, n, p=0.5, alternative="two-sided").pvalue
            records.append({"q": q, "b": b, "n": n, "pct": 100 * b / n, "p_raw": p_raw})

        _, p_adj, _, _ = multipletests([r["p_raw"] for r in records], alpha=0.05, method="fdr_bh")
        for i, r in enumerate(records):
            r["p_adj"] = p_adj[i]
            r["sig"] = p_adj[i] < 0.05

        print(f"\n  Binomial tests by question (FDR-corrected):")
        print(f"  {'Q':<5} {'%B':>6} {'p_adj':>9}  verdict")
        print("  " + "-" * 42)
        for r in records:
            verdict = ("structural ***" if r["pct"] > 50 else "incremental ***") if r["sig"] else "not significant"
            print(f"  {r['q']:<5} {r['pct']:>5.1f}% {r['p_adj']:>9.4f}  {verdict}")

        # Persona + language breakdown
        print(f"\n  % structural by persona:")
        for persona in ["neutral", "ipcc", "unep", "worldbank"]:
            sub = mdf[mdf["persona"] == persona]
            if len(sub) > 0:
                print(f"    {persona:<12} {100*sub['structural'].mean():.1f}%  (n={len(sub):,})")

        print(f"\n  % structural by language:")
        for lang in ["en", "pt"]:
            sub = mdf[mdf["language"] == lang]
            if len(sub) > 0:
                print(f"    {lang:<6} {100*sub['structural'].mean():.1f}%  (n={len(sub):,})")

        # Logistic regression
        print(f"\n  Logistic regression (ref: persona=neutral, lang=en, temp=0.0):")
        try:
            lm = smf.logit(
                "structural ~ C(persona, Treatment('neutral')) "
                "+ C(language, Treatment('en')) "
                "+ C(temp_factor, Treatment('0.0')) "
                "+ C(question_id)",
                data=mdf
            ).fit(disp=False)
            params, pvalues, conf = lm.params, lm.pvalues, lm.conf_int()
            print(f"  {'Term':<40} {'OR':>7} {'95% CI':>18} {'p':>8}")
            print("  " + "-" * 78)
            for name in params.index:
                if name == "Intercept" or name.startswith("C(question_id)"):
                    continue
                p = pvalues[name]
                or_ = np.exp(params[name])
                lo = np.exp(conf.loc[name, 0])
                hi = np.exp(conf.loc[name, 1])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
                ci_str = f"[{lo:.2f}, {hi:.2f}]"
                clean = (name
                    .replace("C(persona, Treatment('neutral'))[T.", "persona=")
                    .replace("C(language, Treatment('en'))[T.", "language=")
                    .replace("C(temp_factor, Treatment('0.0'))[T.", "temp=")
                    .replace("]", ""))
                print(f"  {clean:<40} {or_:>7.3f} {ci_str:>18} {p:>7.4f} {sig}")
            print(f"\n  Pseudo-R2: {lm.prsquared:.4f}  |  N: {int(lm.nobs):,}")
        except Exception as e:
            print(f"  Regression failed: {e}")


# ---------------------------------------------------------------------------
# Analysis 5: PCA over respondent profiles (model x persona x lang x temp x rep)
# ---------------------------------------------------------------------------

# Short labels for questions in plots
Q_LABELS = {
    "Q01": "Q01 transition\nmechanism",
    "Q02": "Q02 externalities",
    "Q03": "Q03 ecosystem\nservices",
    "Q04": "Q04 transition\nscope",
    "Q05": "Q05 policy\nneutrality",
    "Q06": "Q06 speed of\naction",
    "Q07": "Q07 Jevons\nparadox",
    "Q08": "Q08 central\nbank",
    "Q09": "Q09 instrument\nmix",
    "Q10": "Q10 global\ngovernance",
    "Q11": "Q11 climate\ndebt",
    "Q12": "Q12 knowledge\nsystems",
    "Q13": "Q13 growth\nparadigm",
    "Q14": "Q14 nature\nvaluation",
}

MODEL_COLORS = {
    "claude":   "#1f77b4",
    "deepseek": "#d62728",
    "gpt-4o":   "#2ca02c",
    "llama":    "#ff7f0e",
    "mistral":  "#9467bd",
    "qwen":     "#8c564b",
    "sabia":    "#e377c2",
}

PERSONA_MARKERS = {
    "neutral":   "o",
    "ipcc":      "s",
    "unep":      "^",
    "worldbank": "D",
}


def analysis_pca(df):
    print()
    print("=" * 80)
    print("ANALYSIS 5 — PCA OVER RESPONDENT PROFILES")
    print("  Unit of analysis: (model x persona x language x temperature x repetition)")
    print("  Variables: binary structural choice on each of the 14 questions")
    print("=" * 80)

    # Build pivot: one row per (model, persona, language, temperature, rep)
    # rep = run index within that cell — derive it via cumcount
    df2 = df.copy()
    df2["rep"] = df2.groupby(
        ["model", "question_id", "persona", "language", "temp_factor"]
    ).cumcount()

    wide = df2.pivot_table(
        index=["model", "persona", "language", "temp_factor", "rep"],
        columns="question_id",
        values="structural",
        aggfunc="first",
    )

    # Keep only rows with all 14 questions answered
    questions_present = [q for q in Q_LABELS if q in wide.columns]
    wide = wide[questions_present].dropna()
    n_respondents = len(wide)
    print(f"\nRespondent profiles (complete): {n_respondents:,}")
    print(f"Questions included: {len(questions_present)}")

    X = wide.values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(6, len(questions_present)))
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_  # shape (n_components, n_questions)

    # Variance explained
    print("\nVariance explained by component:")
    cumvar = 0.0
    for i, v in enumerate(pca.explained_variance_ratio_):
        cumvar += v
        print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    # PC loadings table
    print("\nQuestion loadings on PC1 and PC2:")
    print(f"  {'Question':<8} {'PC1':>8} {'PC2':>8}  Interpretation")
    print("  " + "-" * 55)
    for j, q in enumerate(questions_present):
        l1, l2 = loadings[0, j], loadings[1, j]
        tag = "irredutible" if q in ("Q05", "Q12") else (
              "contested"   if q in ("Q10", "Q13") else "")
        print(f"  {q:<8} {l1:>8.3f} {l2:>8.3f}  {tag}")

    # Mean PC1/PC2 per model
    score_df = wide.reset_index()[["model", "persona", "language", "temp_factor"]].copy()
    score_df["PC1"] = scores[:, 0]
    score_df["PC2"] = scores[:, 1]

    print("\nMean PC1 score by model (higher = more structural on dominant axis):")
    model_means = score_df.groupby("model")["PC1"].mean().sort_values(ascending=False)
    for m, v in model_means.items():
        print(f"  {m:<12} {v:>7.3f}")

    # --- Figures ---
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: biplot — clip to 5th–95th percentile of scores to exclude outliers
    p5_x, p95_x = np.percentile(scores[:, 0], [3, 97])
    p5_y, p95_y = np.percentile(scores[:, 1], [3, 97])
    pad_x = (p95_x - p5_x) * 0.25
    pad_y = (p95_y - p5_y) * 0.25
    xlim = (p5_x - pad_x, p95_x + pad_x)
    ylim = (p5_y - pad_y, p95_y + pad_y)

    # Scale arrows to fit within clipped window
    arrow_scale = min(abs(xlim[1]), abs(ylim[1])) * 0.75

    fig, ax = plt.subplots(figsize=(11, 8))

    # Respondent points (only those inside clip window)
    mask = (
        (scores[:, 0] >= xlim[0]) & (scores[:, 0] <= xlim[1]) &
        (scores[:, 1] >= ylim[0]) & (scores[:, 1] <= ylim[1])
    )
    for model, grp in score_df.groupby("model"):
        idx = grp.index[mask[grp.index]]
        if len(idx) == 0:
            continue
        color = MODEL_COLORS.get(model, "#333333")
        ax.scatter(scores[idx, 0], scores[idx, 1],
                   c=color, alpha=0.22, s=14, zorder=2)

    # Question loading arrows
    for j, q in enumerate(questions_present):
        l1 = loadings[0, j] * arrow_scale
        l2 = loadings[1, j] * arrow_scale
        # colour irredutible questions differently
        arrow_color = "#c0392b" if q in ("Q05", "Q12") else (
                      "#7f8c8d" if q in ("Q10", "Q13") else "#2c3e50")
        ax.annotate("", xy=(l1, l2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.4))
        short = Q_LABELS.get(q, q).replace("\n", " ")
        ax.text(l1 * 1.15, l2 * 1.15, short,
                ha="center", va="center", fontsize=7.5, color=arrow_color)

    # Model centroids (shown regardless of clipping)
    for model, grp in score_df.groupby("model"):
        cx, cy = grp["PC1"].mean(), grp["PC2"].mean()
        color = MODEL_COLORS.get(model, "#333333")
        ax.scatter(cx, cy, c=color, s=140, edgecolors="white",
                   linewidths=1.5, zorder=6)
        nudge = (p95_y - p5_y) * 0.04
        ax.text(cx, cy + nudge, model, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var — general structural level)",
                  fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var — irredutible vs. contested)",
                  fontsize=11)
    ax.set_title("PCA biplot: respondent profiles × question loadings\n"
                 "Red arrows = irredutible questions (Q05, Q12) · Grey = contested (Q10, Q13)\n"
                 "(3–97th percentile window shown; centroids for all respondents)",
                 fontsize=10)

    legend_patches = [mpatches.Patch(color=c, label=m)
                      for m, c in MODEL_COLORS.items()
                      if m in score_df["model"].unique()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              title="Model", title_fontsize=8)

    plt.tight_layout()
    out_biplot = fig_dir / "pca_biplot.png"
    fig.savefig(out_biplot, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_biplot}")

    # Figure B: PC1 mean per model (bar chart) with persona breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    model_order = model_means.index.tolist()
    persona_order = ["neutral", "ipcc", "unep", "worldbank"]
    x = np.arange(len(model_order))
    width = 0.18
    persona_colors = {"neutral": "#aec7e8", "ipcc": "#ffbb78",
                      "unep": "#98df8a", "worldbank": "#ff9896"}

    for i, persona in enumerate(persona_order):
        vals = [
            score_df[(score_df["model"] == m) & (score_df["persona"] == persona)]["PC1"].mean()
            for m in model_order
        ]
        ax.bar(x + i * width - 1.5 * width, vals, width,
               label=persona, color=persona_colors[persona], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=10)
    ax.set_ylabel("Mean PC1 score", fontsize=11)
    ax.set_title("PC1 score by model and persona\n(higher = more structural on dominant axis)",
                 fontsize=11)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.legend(title="Persona", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    out_bar = fig_dir / "pca_pc1_by_model_persona.png"
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_bar}")

    # Figure C: question correlation heatmap
    corr = pd.DataFrame(X, columns=questions_present).corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(questions_present)))
    ax.set_yticks(range(len(questions_present)))
    ax.set_xticklabels(questions_present, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(questions_present, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Question-level correlation matrix\n"
                 "(computed across all respondent profiles)", fontsize=11)
    plt.tight_layout()
    out_corr = fig_dir / "pca_question_correlation.png"
    fig.savefig(out_corr, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_corr}")


# ---------------------------------------------------------------------------
# Analysis 6: Variance decomposition — ICC per grouping factor
# ---------------------------------------------------------------------------

def analysis_icc(df):
    """
    Empirical ICC for each grouping factor (model, question, persona, language,
    temperature).

    Method: one-way ANOVA decomposition on the raw binary outcome.

    NOTE ON QUESTION AS A VARIABLE:
    "Question" is the content of the stimulus, not an experimental manipulation.
    It is excluded from the primary ICC decomposition, which focuses on the four
    factors the researcher actually varied: model, persona, language, temperature.
    Question heterogeneity is instead used as the denominator for a second analysis:
    the within-question ICC, which asks "of the variance that question content does
    NOT explain, how much do the experimental factors explain?"

    Part A — ICC of experimental factors (model, persona, language, temperature)
              relative to total outcome variance.
    Part B — Within-question ICC: same factors relative to residual variance after
              removing question-level means (i.e., variance unexplained by content).
    Part C — Cross-validation via incremental pseudo-R2 (logistic, question FE).
    """
    print()
    print("=" * 80)
    print("ANALYSIS 6 -- VARIANCE DECOMPOSITION (ICC)")
    print("  Question is treated as content context, not an experimental variable.")
    print("  Part A: ICC relative to total variance.")
    print("  Part B: Within-question ICC (variance not explained by question content).")
    print("=" * 80)

    p_overall = df["structural"].mean()
    var_total = p_overall * (1 - p_overall)
    n_total = len(df)

    print(f"\nOverall P(structural): {p_overall*100:.1f}%")
    print(f"Total Bernoulli variance: {var_total:.4f}  (N = {n_total:,})")

    # ------------------------------------------------------------------ #
    # Helper: weighted between-group variance for a column                #
    # ------------------------------------------------------------------ #
    def _between_var(col):
        grp = df.groupby(col)["structural"]
        means = grp.mean()
        sizes = grp.count()
        return float(np.average((means - p_overall) ** 2, weights=sizes))

    # ------------------------------------------------------------------ #
    # Part A: ICC relative to total variance (experimental factors only)  #
    # ------------------------------------------------------------------ #
    exp_factors = {
        "model":       "model",
        "persona":     "persona",
        "language":    "language",
        "temperature": "temp_factor",
    }

    print()
    print("PART A -- ICC of experimental factors (denominator = total variance)")
    print(f"  {'Factor':<14} {'Levels':>6}  {'ICC / total':>12}  {'Between var':>13}")
    print("  " + "-" * 52)

    icc_total = {}
    for label, col in exp_factors.items():
        bv = _between_var(col)
        icc = bv / var_total
        icc_total[label] = {"icc": icc, "between_var": bv,
                             "n_levels": df[col].nunique()}
        print(f"  {label:<14} {df[col].nunique():>6}  {icc*100:>11.2f}%  {bv:>13.5f}")

    # ------------------------------------------------------------------ #
    # Part B: within-question ICC                                          #
    # residual variance = total variance minus question between-group var  #
    # ------------------------------------------------------------------ #
    bv_question = _between_var("question_id")
    var_within_q = var_total - bv_question
    pct_question = bv_question / var_total * 100

    print()
    print("PART B -- Within-question ICC (denominator = variance not explained by question)")
    print(f"  Question content accounts for {pct_question:.1f}% of total variance by design.")
    print(f"  Remaining (within-question) variance: {var_within_q:.4f} "
          f"({100 - pct_question:.1f}% of total)")
    print()
    print(f"  {'Factor':<14} {'Levels':>6}  {'ICC / within-Q':>15}  {'ICC / total':>12}")
    print("  " + "-" * 56)

    icc_within = {}
    for label, col in exp_factors.items():
        bv = icc_total[label]["between_var"]
        icc_w = bv / var_within_q
        icc_within[label] = icc_w
        icc_t = icc_total[label]["icc"]
        print(f"  {label:<14} {df[col].nunique():>6}  {icc_w*100:>14.2f}%  {icc_t*100:>11.2f}%")

    # ------------------------------------------------------------------ #
    # Part C: cross-validation via incremental pseudo-R2                  #
    # ------------------------------------------------------------------ #
    print()
    print("PART C -- Cross-validation: incremental pseudo-R2 (logistic, question FE)")
    print(f"  {'Factor added':<18} {'dPseudo-R2':>12}  {'Pseudo-R2 total':>17}")
    print("  " + "-" * 52)

    base_formula = "structural ~ C(question_id)"
    try:
        m0 = smf.logit(base_formula, data=df).fit(disp=False, maxiter=200)
        r2_base = m0.prsquared
        print(f"  {'baseline (Q FE)':<18} {'---':>12}  {r2_base*100:>16.2f}%")

        factor_formulas = {
            "model":       "C(model, Treatment('claude'))",
            "persona":     "C(persona, Treatment('neutral'))",
            "language":    "C(language, Treatment('en'))",
            "temperature": "C(temp_factor, Treatment('0.0'))",
        }
        for label, term in factor_formulas.items():
            formula = f"{base_formula} + {term}"
            m = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
            delta = m.prsquared - r2_base
            print(f"  {('+ ' + label):<18} {delta*100:>11.2f}%  {m.prsquared*100:>16.2f}%")
    except Exception as e:
        print(f"  Logistic cross-validation failed: {e}")

    # Summary bar
    print()
    print("Summary ranking (within-question ICC):")
    sorted_w = sorted(icc_within.items(), key=lambda x: x[1], reverse=True)
    for label, val in sorted_w:
        bar = "#" * int(val * 500)
        print(f"  {label:<14} {val*100:5.2f}%  {bar}")

    # ------------------------------------------------------------------ #
    # Figure: two-panel stacked bar                                        #
    # ------------------------------------------------------------------ #
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    factor_colors = {
        "model":       "#d7191c",
        "persona":     "#fdae61",
        "language":    "#abd9e9",
        "temperature": "#d9d9d9",
    }
    exp_order = sorted(icc_total.keys(),
                       key=lambda l: icc_total[l]["icc"], reverse=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={"hspace": 1.1})

    for ax, (mode, icc_dict, denom_label, denom_val) in zip(
        axes,
        [
            ("Part A: % of total outcome variance",
             {l: icc_total[l]["icc"] * 100 for l in exp_order},
             "question content + residual",
             100 - sum(icc_total[l]["icc"] * 100 for l in exp_order)),
            ("Part B: % of within-question variance\n(after removing question-content effect)",
             {l: icc_within[l] * 100 for l in exp_order},
             "residual",
             100 - sum(icc_within[l] * 100 for l in exp_order)),
        ],
    ):
        left = 0
        for label in exp_order:
            val = icc_dict[label]
            color = factor_colors[label]
            ax.barh(0, val, left=left, height=0.5, color=color,
                    label=f"{label} ({val:.1f}%)", edgecolor="white")
            if val > 0.8:
                ax.text(left + val / 2, 0, f"{val:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if val > 5 else "black",
                        fontweight="bold")
            left += val

        residual = max(0.0, denom_val)
        ax.barh(0, residual, left=left, height=0.5, color="#eeeeee",
                label=f"{denom_label} ({residual:.1f}%)", edgecolor="white")

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of variance", fontsize=10)
        ax.set_title(mode, fontsize=10, loc="left")
        ax.legend(loc="upper right", bbox_to_anchor=(1, -0.35),
                  ncol=3, fontsize=8, frameon=False)

    fig.suptitle("Variance decomposition (ICC) — experimental factors only\n"
                 "Question treated as content context, not experimental variable",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_icc = fig_dir / "icc_variance_decomposition.png"
    fig.savefig(out_icc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_icc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    tee = _Tee(OUT_PATH)
    sys.stdout = tee

    try:
        conn = sqlite3.connect(DB_PATH)
        df = load_data(conn)
        conn.close()

        n_models = df["model_name"].nunique()
        n_obs = len(df)
        models_in_db = sorted(df["model_name"].unique())
        print(f"Data loaded: {n_obs:,} valid observations, {n_models} models")
        print(f"Models: {', '.join(short_name(m) for m in models_in_db)}")

        analysis_binomial(df)
        analysis_regression(df)
        analysis_dimensions(df)
        analysis_per_model(df)
        analysis_pca(df)
        analysis_icc(df)

        print(f"\nOutput saved to: {OUT_PATH}")
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
