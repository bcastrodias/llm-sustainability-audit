# Measuring Normative Preferences in Sustainability Policy: A Multi-Model Audit of Large Language Models — Data and Code

Dataset and analysis code for the article *Measuring Normative Preferences in Sustainability Policy: A Multi-Model Audit of Large Language Models* (Fonseca & Klink).

The study audits seven large language models on the economic policies advanced to drive decarbonisation and ecological transition, using a 14-item binary forced-choice questionnaire under a full factorial design (institutional persona × language × temperature × presentation order × repetitions). Data were collected between 1 and 3 May 2026.

## Contents

```
data/experiment.db              SQLite database with the full experiment (raw)
src/                            Experiment engine: config, database, model clients, runner
main.py, run_sabia.py           Data-collection entry points
analyze.py                      Analysis: binomial tests + FDR, logistic regressions,
                                ICC variance decomposition, PCA; writes results/stats_output.txt
make_results_heatmap.py         Heatmaps of structural choices by model and question
make_persona_language_plots.py  Persona forest plot and per-model language odds ratios
make_variance_decomp.py         Variance-decomposition figure
results/stats_output.txt        Full statistical output (the authoritative numbers)
results/figures/                The five figures used in the paper
requirements.txt                Python dependencies
.env.example                    Template for the API keys needed to re-collect data
```

## Dataset (`data/experiment.db`)

- **`runs`** (31,360 rows; 31,196 with a valid A/B answer) — one row per model call:
  `question_id, model_name, order_ab, persona, language, temperature, rep_number, choice,
  justification, raw_response, tokens_input, tokens_output, cost_usd, latency_ms, created_at`.
  `choice` is normalised so that **A = incremental** and **B = structural** regardless of the
  presentation order; values other than A/B (`refused`, `error`) are excluded from analysis.
- **`questions`** — the 14 items: `id, theme, dimension, incremental_is_a,
  pole_a_en, pole_b_en, pole_a_pt, pole_b_pt` (also defined in `src/db.py`).
- **`models`** — the seven audited models: `name, provider, model_id, api_base`.
- **`api_health`** — health-check log (no analytical content).

## Reproducing the analysis

```bash
pip install -r requirements.txt
python analyze.py                       # data/experiment.db -> results/stats_output.txt and figures
python make_results_heatmap.py
python make_persona_language_plots.py
python make_variance_decomp.py
```

All scripts read `data/experiment.db` and are run from the repository root.

Re-collecting the data (optional) requires your own API keys:

```bash
cp .env.example .env                    # then fill in the keys
python main.py pilot                    # or run_sabia.py for the Sabiá-3 runner
```

No API keys are included in this deposit; collection is only needed to regenerate the database.

## Dependencies
Python 3.11+. See `requirements.txt`: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn and scienceplots for analysis and figures; anthropic, openai and python-dotenv for collection.

## Licence (suggested)
Code: MIT. Data: CC-BY-4.0.

## Citation
Fonseca, B. C. D., & Klink, J. (2026). *Measuring Normative Preferences in Sustainability Policy: A Multi-Model Audit of Large Language Models* [Data and code]. Zenodo. https://doi.org/XXXXX
