# SAFER Road Severity Lab

Compact Streamlit project for exploring three synthetic pre-crash scenario datasets, estimating Slight/Serious/Fatal severity conditional on a hypothetical collision, comparing models and calibration, and reviewing local/global explanations and scenario differences.

## Run from PowerShell

Open PowerShell in this extracted folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run app.py
```

The package includes the fitted model and all three datasets, so you do not need to run training first. Use Python 3.13 for the pinned dependencies and fitted model artifact.

## Included files

- `app.py`, `app_pages/` — Streamlit app pages.
- `src/generate_synthetic_datasets.py` — reproducibly regenerates A and B with collision-count fields excluded.
- `src/generate_dataset_c.py` — creates the severe-imbalance prior-shift sample from B.
- `src/train_model.py`, `src/model_service.py`, `src/postfit_diagnostics.py` — training, inference, local SHAP, permutation importance, and label-rule sensitivity.
- `data/synthetic_dataset_A_10000.csv` — Dataset A development and final test partitions.
- `data/synthetic_dataset_B_5000.csv` — independent draw from the same synthetic simulator.
- `data/synthetic_dataset_C_prior_shift_2000.csv` — 1,503 Slight / 470 Serious / 27 Fatal rows sampled from B without replacement; it overlaps B by design.
- `data/synthetic_generation_specification.json`, `data/synthetic_context_marginals.json`, `data/risk_mappings.csv` — frozen simulator assumptions and synthetic context distributions.
- `outputs/ensemble_comparison.csv` — base learners and uniform/adaptive ensemble results on A, B, and C.
- `outputs/paired_bootstrap_comparisons.csv` — 1,000 paired bootstrap comparisons against uniform soft voting.
- `outputs/calibration_metrics.csv`, `outputs/reliability_diagram_points.csv` — Brier score, log loss, expected calibration error, and reliability-curve values.
- `outputs/permutation_importance.csv` — global test-set input permutation associations.
- `outputs/simulator_sensitivity.csv` — frozen-model agreement with alternative synthetic weight/threshold rules.
- `data/class_report_*.csv`, `data/confusion_*.csv` — class reports and confusion matrices.
- `trained_model/safer_raae.joblib` — fitted adaptive ensemble.
- `docs/SAFER_Final_Report.md` — updated IEEE-style report content with the feature-availability audit, coefficient rationale, revised results, comparison intervals, and limits.
- `docs/MODEL_CARD.md` — model scope, intended use, evaluation, and limitations.
- `outputs/interactive_model_metrics.json` — metrics from the included model refit.
- `requirements.txt` — pinned Python dependencies.
- `DEPLOYMENT.md` — local and internet hosting notes.

## Important scope

All records, pre-impact speeds, and severity targets are synthetic. Context marginals are frozen from an earlier synthetic Dataset A; the generator samples context fields independently, so it does not represent real-world joint distributions. The scenario form asks for 13 conditions; number of vehicles is removed because it is generally post-collision information. Actual speed, speed excess, and speeding are hypothetical simulator values, not observations. Dataset B uses the same simulator and C is a subset of B; neither provides real-world external validation. Calibration behavior changes by metric and sample. Scores are not calibrated real-world probabilities. Severity is conditional on a hypothetical collision, not a crash-occurrence estimate. The adaptive ensemble is not claimed superior without comparative evidence. See `EXPERIMENT_UPDATE.md` for results and limitations.

## Reproduce the experiment

```powershell
python src/generate_synthetic_datasets.py
python src/generate_dataset_c.py
python src/train_model.py
```

The simulator coefficients, speed-offset distribution, class thresholds, and Gaussian noise are transparent design assumptions. `simulator_noise`, `simulator_risk_score`, and `severity` are audit/target-generation columns and are excluded from model features.
