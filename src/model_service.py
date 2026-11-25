"""Prediction and local explanation services for the SAFER demo model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "trained_model" / "safer_raae.joblib"
LABELS = ["Slight", "Serious", "Fatal"]
FEATURES = [
    "day_type", "hour", "speed_limit_kmh", "road_type", "junction_detail",
    "junction_control", "urban_or_rural", "weather_conditions", "light_conditions",
    "road_surface_conditions", "vehicle_type_involved", "driver_age_band",
    "actual_speed_kmh_simulated", "speed_excess_kmh_simulated",
    "speeding_indicator_simulated",
]
NUMERIC_FEATURES = ["hour", "speed_limit_kmh", "actual_speed_kmh_simulated", "speed_excess_kmh_simulated", "speeding_indicator_simulated"]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]


class SaferPredictor:
    """Entropy-weighted soft-voting model plus raw-feature SHAP explanations."""

    def __init__(self, artifact: dict[str, Any], background: pd.DataFrame):
        self.models: dict[str, Any] = artifact["models"]
        self.alpha: float = float(artifact["alpha"])
        self.beta: float = float(artifact.get("beta", 0.0))
        self.calibrator: dict[str, Any] = artifact.get("calibrator", {"method": "raw"})
        self.calibration_method: str = str(artifact.get("calibration_method", "raw"))
        self.fatal_threshold: float = float(artifact["fatal_threshold"])
        self.labels: list[str] = list(artifact["labels"])
        self.background = background[FEATURES].copy()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return adaptive weighted class scores ordered as LABELS."""
        frame = frame[FEATURES].copy()
        per_model = []
        reliabilities = []
        for estimator in self.models.values():
            raw = estimator.predict_proba(frame)
            model_classes = list(estimator.named_steps["model"].classes_)
            aligned = np.column_stack([raw[:, model_classes.index(label)] for label in LABELS])
            entropy = -(aligned * np.log(np.clip(aligned, 1e-12, 1.0))).sum(axis=1) / np.log(len(LABELS))
            reliability = np.exp(-self.alpha * entropy)
            per_model.append(aligned)
            reliabilities.append(reliability)
        stack = np.stack(per_model, axis=1)
        consensus = np.mean(stack, axis=1, keepdims=True)
        disagreement = (
            stack * np.log(np.clip(stack, 1e-12, 1.0) / np.clip(consensus, 1e-12, 1.0))
        ).sum(axis=2) / np.log(len(LABELS))
        weights = np.stack(reliabilities, axis=1) * np.exp(-self.beta * disagreement)
        weights /= weights.sum(axis=1, keepdims=True)
        probabilities = np.zeros_like(per_model[0])
        for idx, p in enumerate(per_model):
            probabilities += weights[:, idx, None] * p
        return self.calibrate(probabilities)

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the selected one-vs-rest calibration mapping and renormalize."""
        method = self.calibrator.get("method", "raw")
        if method == "raw":
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        calibrated = []
        for index, model in enumerate(self.calibrator["models"]):
            if method == "platt":
                calibrated.append(expit(model["coef"] * probabilities[:, index] + model["intercept"]))
            elif method == "isotonic":
                calibrated.append(model.predict(probabilities[:, index]))
            else:
                raise ValueError(f"Unsupported stored calibration method: {method}")
        calibrated = np.column_stack(calibrated).clip(1e-9, 1.0)
        return calibrated / calibrated.sum(axis=1, keepdims=True)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(frame)
        output = []
        for row in probabilities:
            if row[LABELS.index("Fatal")] >= self.fatal_threshold:
                output.append("Fatal")
            else:
                nonfatal = [LABELS.index("Slight"), LABELS.index("Serious")]
                output.append(LABELS[nonfatal[int(np.argmax(row[nonfatal]))]])
        return np.asarray(output, dtype=object)

    def explain(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Permutation SHAP for the adaptive probability function.

        SHAP's independent tabular masker expects numeric inputs, so explain
        the common one-hot encoded representation and then sum each encoded
        field's signed attribution back to its original input column.
        """
        first_pipe = next(iter(self.models.values()))
        preprocessor = first_pipe.named_steps["preprocess"]

        def dense(values):
            return values.toarray() if sparse.issparse(values) else np.asarray(values)

        background = dense(preprocessor.transform(self.background)).astype(float)
        target = dense(preprocessor.transform(frame[FEATURES].iloc[:1])).astype(float)
        encoded_names = list(preprocessor.get_feature_names_out())

        def score_encoded(matrix: np.ndarray) -> np.ndarray:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            model_probabilities = []
            reliabilities = []
            for estimator in self.models.values():
                raw = estimator.named_steps["model"].predict_proba(matrix)
                classes = list(estimator.named_steps["model"].classes_)
                aligned = np.column_stack([raw[:, classes.index(label)] for label in LABELS])
                entropy = -(aligned * np.log(np.clip(aligned, 1e-12, 1.0))).sum(axis=1) / np.log(len(LABELS))
                model_probabilities.append(aligned)
                reliabilities.append(np.exp(-self.alpha * entropy))
            stack = np.stack(model_probabilities, axis=1)
            consensus = np.mean(stack, axis=1, keepdims=True)
            disagreement = (
                stack * np.log(np.clip(stack, 1e-12, 1.0) / np.clip(consensus, 1e-12, 1.0))
            ).sum(axis=2) / np.log(len(LABELS))
            weights = np.stack(reliabilities, axis=1) * np.exp(-self.beta * disagreement)
            weights /= weights.sum(axis=1, keepdims=True)
            combined = np.zeros_like(model_probabilities[0])
            for index, probabilities in enumerate(model_probabilities):
                combined += weights[:, index, None] * probabilities
            return self.calibrate(combined)

        explainer = shap.PermutationExplainer(
            score_encoded,
            shap.maskers.Independent(background, max_samples=len(background)),
            feature_names=encoded_names,
        )
        explanation = explainer(target, max_evals=2 * len(encoded_names) + 1)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        if values.ndim == 2:
            values = values[:, :, None]
        encoded_values = values[0]
        grouped = np.zeros((len(FEATURES), encoded_values.shape[-1]), dtype=float)
        for encoded_index, encoded_name in enumerate(encoded_names):
            source_name = encoded_name.split("__", 1)[-1]
            matches = [name for name in FEATURES if source_name == name or source_name.startswith(name + "_")]
            if matches:
                feature_index = FEATURES.index(max(matches, key=len))
                grouped[feature_index] += encoded_values[encoded_index]
        return grouped, np.ravel(base[0])


def load_predictor() -> SaferPredictor:
    """Load the fitted model and deterministic explanation background."""
    artifact = joblib.load(MODEL_PATH)
    background = pd.DataFrame(artifact["explanation_background"], columns=FEATURES)
    return SaferPredictor(artifact, background)

