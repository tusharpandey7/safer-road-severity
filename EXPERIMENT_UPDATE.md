# SAFER pre-crash schema and experiment update

## Research question and contribution

SAFER estimates collision severity conditional on a hypothetical collision having occurred. It does not estimate whether a collision will occur. Its defensible contribution is an integrated and reproducible evaluation of per-observation entropy-adaptive ensemble weighting, validation-selected probability calibration and Fatal decision threshold, local SHAP explanations, and performance under a synthetic low-Fatal-prior stress sample. Logistic Regression, Random Forest, LightGBM, entropy weighting, calibration, thresholding, permutation importance, and SHAP are established methods; SAFER does not claim to introduce a new classifier. The test results do not establish that the adaptive ensemble is consistently superior to uniform soft voting.

## Data-availability changes

The pre-crash formulation removes `number_of_vehicles`, which generally becomes known only after a collision. Posted speed limit remains a separate legal road attribute. The revised simulator adds `actual_speed_kmh_simulated`, `speed_excess_kmh_simulated`, and `speeding_indicator_simulated`. These are hypothetical simulator features, not observations from a speed camera, telematics, police investigation, or API. The Streamlit scenario form asks the user for posted limit and hypothetical speed; it derives speed excess and the indicator. The fitted model has 15 predictors, including two engineered speed fields. It does not use `collision_date`, `simulator_noise`, `simulator_risk_score`, or the target `severity` as predictors.

The synthetic context marginals are frozen from the earlier synthetic Dataset A and stored with the project. Context fields are sampled independently from those marginals, except that `day_type` is derived from the generated date and simulated vehicle speed is derived from posted limit plus a declared random offset. These choices make the generator reproducible; they do not reproduce measured real-world joint distributions and can create combinations that would need plausibility review.

## Synthetic label rule

For each contextual variable, the generator maps a category or numeric range to a normalized design score in [0,1]. The base score is

`R_base = .18 r_limit + .16 r_vehicle + .08 r_road + .08 r_junction + .04 r_control + .04 r_area + .08 r_weather + .08 r_light + .08 r_surface + .06 r_age + .04 r_hour + .08 r_speed_excess`.

The interaction score is

`I = .10(r_speed_excess r_vehicle) + .08(r_weather r_surface) + .06(r_light r_hour)`.

The final score is `R = clip(R_base + I + epsilon, 0, 1.35)`, where `epsilon ~ Normal(0, 0.018^2)`. Labels use fixed thresholds: Slight below 0.299, Serious from 0.299 to below 0.393, and Fatal from 0.393 onward. The weights, interaction strengths, synthetic speed-offset distribution, noise, and thresholds are transparent design assumptions informed by qualitative road-safety reasoning. They were not estimated with real crash data, expert elicitation, or a causal study, and must not be presented as scientific constants.

| Design parameter | Value | Rationale and limit |
|---|---:|---|
| Posted speed limit | 0.18 | Higher relative weight for road-speed context; not the speed driven. |
| Vehicle type | 0.16 | Represents designed differences among vehicle categories. |
| Road type / junction detail | 0.08 each | Represents layout context in the simulator. |
| Junction control / area | 0.04 each | Smaller contextual contributions by design. |
| Weather / lighting / surface | 0.08 each | Represents environmental and visibility conditions. |
| Driver age band | 0.06 | Categorical simulator contribution. |
| Hour | 0.04 | Time-of-day contribution. |
| Speed excess | 0.08 | Synthetic replacement for post-collision vehicle count. |
| Speed excess × vehicle type | 0.10 | Designed compounding interaction. |
| Weather × surface | 0.08 | Designed compounding interaction. |
| Lighting × hour | 0.06 | Designed time/visibility interaction. |
| Gaussian noise | SD 0.018 | Adds controlled variation; not estimated measurement error. |

## Datasets and model protocol

| Dataset | Rows | Slight | Serious | Fatal | Use |
|---|---:|---:|---:|---:|---|
| A | 10,000 | 3,778 | 3,888 | 2,334 | 60% fit, 10% calibration, 10% validation, 20% untouched test |
| B | 5,000 | 1,905 | 1,977 | 1,118 | Independent draw from the same simulator; evaluation only |
| C | 2,000 | 1,503 | 470 | 27 | Severe class-prior stress sample drawn from B without replacement |

Dataset C has 1.35% Fatal rows, the nearest lower whole-row representation to a requested 1.37% prior at 2,000 rows. It overlaps B by construction, so B and C are not independent replications. C has only 27 Fatal observations; its precision and recall are highly uncertain. All datasets are synthetic.

The three class-weighted base learners are Logistic Regression, Random Forest, and LightGBM. Uniform soft voting is compared with entropy-adaptive weighting and entropy plus disagreement penalties. Beta is selected from {0, 0.5, 1, 2} on validation Macro-F1. One-vs-rest Platt and isotonic calibration are fitted on the separate calibration partition; method selection and the Fatal threshold use validation only. The final run selected beta=2, isotonic calibration, and a Fatal alert threshold of 0.40. Paired 1,000-resample bootstrap intervals compare each configuration with uniform voting on the same evaluation rows.

## Classification results

| Evaluation | Accuracy | Balanced accuracy | Macro-F1 | Fatal precision | Fatal recall |
|---|---:|---:|---:|---:|---:|
| Dataset A untouched test | 84.75% | 84.91% | 84.98% | 85.68% | 85.87% |
| Dataset B same-simulator | 84.68% | 84.80% | 84.89% | 85.42% | 85.42% |
| Dataset C low-Fatal stress | 85.65% | 84.01% | 72.18% | 37.93% | 81.48% |

On Dataset C, high accuracy coexists with lower Macro-F1 and Fatal precision because Fatal is only 1.35% of the sample. The validation-selected Fatal threshold is a sensitivity-oriented alert rule, not an ordinary argmax class decision. Report the three class scores, the selected class, and this threshold behavior separately.

On the untouched A test set, uniform soft voting achieved 84.50% accuracy and 84.67% Macro-F1. The selected beta=2 configuration achieved 84.50% and 84.61%. The paired 95% bootstrap intervals for beta=2 minus uniform were [-0.35, +0.35] percentage points for accuracy, [-0.39, +0.36] points for balanced accuracy, and [-0.43, +0.31] points for Macro-F1. Each interval spans zero; this run does not establish a reliable improvement from disagreement weighting.

## Calibration

| Dataset A test method | Multiclass Brier (lower better) | Log loss (lower better) | Macro OVR ECE (lower better) |
|---|---:|---:|---:|
| Raw | 0.2199 | 0.3699 | 0.0498 |
| Platt | 0.2127 | 0.3576 | 0.0333 |
| Isotonic (selected on validation) | 0.2129 | 0.3471 | 0.0202 |

On Dataset C, raw / Platt / isotonic Brier scores were 0.1982 / 0.1997 / 0.2026; log losses were 0.3327 / 0.3423 / 0.3230; and macro OVR ECE values were 0.0727 / 0.0763 / 0.0695. Calibration behavior changes with the sample and metric. These figures describe synthetic labels and do not establish probability calibration for actual crashes.

## Explainability and sensitivity

Local permutation SHAP is available for each Streamlit scenario. Global permutation importance is computed on the untouched Dataset A test set over five repeats. The leading Macro-F1 drops were vehicle type (0.2648), posted speed limit (0.1864), lighting (0.0935), simulated speed excess (0.0885), and weather (0.0762). These measure model sensitivity to shuffled synthetic inputs; they are not causal effects.

The project also varies major simulator weights and class thresholds by ±25% or ±0.02 while holding the model fixed. This reports agreement with alternative simulator labels rather than retrained-model performance. For example, increasing the Fatal threshold from 0.393 to 0.413 changed 6.15% of held-out labels; the frozen-model agreement accuracy was 83.40% and Fatal recall was 94.77%. Lowering the threshold to 0.373 changed 7.05% of labels, with 81.80% agreement accuracy and 72.70% Fatal recall. The Streamlit page shows all tested variants and emphasizes this scope.

The scenario comparison dashboard contrasts two hypothetical inputs and displays changes in class scores. This is a model comparison only. It does not estimate what would causally happen if a real driver or road condition changed.

## Real-data boundary and future work

The earlier Indian highway dataset was useful as a data-quality case study, and STATS19 demonstrated how class imbalance affects collision-severity classification. That prior STATS19 experiment included `number_of_vehicles`, a field that is generally post-collision. Its metrics therefore are not evidence for a pre-crash model. The current package contains no independent real-world pre-crash validation. Before making real-world claims, SAFER needs a permitted dataset with only pre-event predictors, measured or defensibly estimated actual speed, leakage review, temporal and geographic external evaluation, calibration under the relevant class prior, and analysis of false-positive and false-negative costs.

## Reproducibility

From the project root, run `python src/generate_synthetic_datasets.py`, then `python src/generate_dataset_c.py`, then `python src/train_model.py`. The frozen context marginals, category mappings, generation specification, seeds, training artifact, metrics, calibration data, permutation results, and sensitivity results are included in the package. The model is a college research prototype and is not intended for operational, emergency-response, insurance, enforcement, or safety-critical decisions.
