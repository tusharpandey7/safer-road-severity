"""Global input importance and simulator-rule sensitivity diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.generate_synthetic_datasets import INTERACTIONS, THRESHOLDS, WEIGHTS, calculate_risk, severity_from_risk
from src.model_service import FEATURES, load_predictor


def write_permutation_importance(x_test: pd.DataFrame, y_test: pd.Series, output_path: Path, repeats: int = 5) -> None:
    predictor = load_predictor()
    baseline_prediction = predictor.predict(x_test)
    baseline = f1_score(y_test, baseline_prediction, labels=["Slight", "Serious", "Fatal"], average="macro", zero_division=0)
    rng = np.random.default_rng(381)
    rows = []
    for feature in FEATURES:
        drops = []
        for _ in range(repeats):
            permuted = x_test.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            prediction = predictor.predict(permuted)
            score = f1_score(y_test, prediction, labels=["Slight", "Serious", "Fatal"], average="macro", zero_division=0)
            drops.append(baseline - score)
        rows.append({
            "Feature": feature,
            "Macro-F1 drop mean": float(np.mean(drops)),
            "Macro-F1 drop sd": float(np.std(drops, ddof=1)) if repeats > 1 else 0.0,
            "Repeats": repeats,
            "Baseline macro-F1": float(baseline),
            "Interpretation": "Synthetic test-set predictive association; not causal importance",
        })
    pd.DataFrame(rows).sort_values("Macro-F1 drop mean", ascending=False).to_csv(output_path, index=False)


def write_rule_sensitivity(x_test: pd.DataFrame, y_test: pd.Series, noise: pd.Series, output_path: Path) -> None:
    predictor = load_predictor()
    frozen_prediction = predictor.predict(x_test)
    base_labels = np.asarray(y_test, dtype=str)
    rows = []
    variants: list[tuple[str, dict[str, float], dict[str, float], tuple[float, float]]] = [
        ("Baseline specification", dict(WEIGHTS), dict(INTERACTIONS), THRESHOLDS),
    ]
    for name in ("speed_limit", "speed_excess"):
        for factor in (0.75, 1.25):
            weights = dict(WEIGHTS)
            weights[name] *= factor
            scale = sum(WEIGHTS.values()) / sum(weights.values())
            weights = {key: value * scale for key, value in weights.items()}
            variants.append((f"{name} weight {factor:.2f}x (renormalized)", weights, dict(INTERACTIONS), THRESHOLDS))
    for name in ("speed_excess_x_vehicle", "weather_x_surface", "light_x_hour"):
        for factor in (0.75, 1.25):
            interactions = dict(INTERACTIONS)
            interactions[name] *= factor
            variants.append((f"{name} interaction {factor:.2f}x", dict(WEIGHTS), interactions, THRESHOLDS))
    for label, thresholds in [
        ("Slight threshold -0.02", (THRESHOLDS[0] - 0.02, THRESHOLDS[1])),
        ("Slight threshold +0.02", (THRESHOLDS[0] + 0.02, THRESHOLDS[1])),
        ("Fatal threshold -0.02", (THRESHOLDS[0], THRESHOLDS[1] - 0.02)),
        ("Fatal threshold +0.02", (THRESHOLDS[0], THRESHOLDS[1] + 0.02)),
    ]:
        variants.append((label, dict(WEIGHTS), dict(INTERACTIONS), thresholds))

    for name, weights, interactions, thresholds in variants:
        risk = calculate_risk(x_test, noise=noise.to_numpy(), weights=weights, interactions=interactions)
        alternative_labels = severity_from_risk(risk, thresholds)
        rows.append({
            "Variant": name,
            "Slight threshold": thresholds[0],
            "Fatal threshold": thresholds[1],
            "Alternative Fatal share": float(np.mean(alternative_labels == "Fatal")),
            "Label changes from baseline": float(np.mean(alternative_labels != base_labels)),
            "Frozen-model agreement accuracy": float(accuracy_score(alternative_labels, frozen_prediction)),
            "Frozen-model agreement macro-F1": float(f1_score(alternative_labels, frozen_prediction, labels=["Slight", "Serious", "Fatal"], average="macro", zero_division=0)),
            "Frozen-model Fatal recall": float(recall_score(alternative_labels == "Fatal", frozen_prediction == "Fatal", zero_division=0)),
            "Interpretation": "Same held-out rows and frozen model; alternate synthetic label rule, not retrained performance",
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
