"""Fit the interactive SAFER model without using Dataset B for training."""
from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_service import CATEGORICAL_FEATURES, FEATURES, LABELS, NUMERIC_FEATURES


SEED = 42
ALPHA = 2.0
FATAL_THRESHOLD = 0.470


def make_pipeline(kind: str) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    if kind == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced", max_iter=2500, C=1.0,
            solver="lbfgs", random_state=SEED,
        )
    elif kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=240, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        )
    elif kind == "lightgbm":
        estimator = LGBMClassifier(
            objective="multiclass", n_estimators=240, learning_rate=0.05,
            num_leaves=24, max_depth=-1, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown estimator: {kind}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def aligned_proba(pipe: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = pipe.predict_proba(frame[FEATURES])
    classes = list(pipe.named_steps["model"].classes_)
    return np.column_stack([raw[:, classes.index(label)] for label in LABELS])


def adaptive_proba(models: dict[str, Pipeline], frame: pd.DataFrame) -> np.ndarray:
    model_probs = [aligned_proba(pipe, frame) for pipe in models.values()]
    reliabilities = []
    for probabilities in model_probs:
        entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1) / np.log(len(LABELS))
        reliabilities.append(np.exp(-ALPHA * entropy))
    weights = np.stack(reliabilities, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    combined = np.zeros_like(model_probs[0])
    for index, probabilities in enumerate(model_probs):
        combined += weights[:, index, None] * probabilities
    return combined


def decide(probabilities: np.ndarray) -> np.ndarray:
    predictions = []
    fatal_idx = LABELS.index("Fatal")
    nonfatal = [LABELS.index("Slight"), LABELS.index("Serious")]
    for row in probabilities:
        if row[fatal_idx] >= FATAL_THRESHOLD:
            predictions.append("Fatal")
        else:
            predictions.append(LABELS[nonfatal[int(np.argmax(row[nonfatal]))]])
    return np.asarray(predictions, dtype=object)


def evaluate(y: pd.Series, probabilities: np.ndarray) -> dict:
    prediction = decide(probabilities)
    report = classification_report(
        y, prediction, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
        "fatal_precision": float(report["Fatal"]["precision"]),
        "fatal_recall": float(report["Fatal"]["recall"]),
        "fatal_f1": float(report["Fatal"]["f1-score"]),
        "fatal_pr_auc": float(average_precision_score((y == "Fatal").astype(int), probabilities[:, LABELS.index("Fatal")])),
        "class_report": report,
        "confusion_matrix": confusion_matrix(y, prediction, labels=LABELS).tolist(),
    }


def main() -> None:
    data_dir = ROOT / "data"
    output_dir = ROOT / "outputs"
    model_dir = ROOT / "trained_model"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    dataset_a = pd.read_csv(data_dir / "synthetic_dataset_A_10000.csv")
    dataset_b = pd.read_csv(data_dir / "synthetic_dataset_B_5000.csv")
    x = dataset_a[FEATURES].copy()
    y = dataset_a["severity"].astype(str)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.40, random_state=SEED, stratify=y
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, random_state=SEED, stratify=y_holdout
    )

    models: dict[str, Pipeline] = {}
    for name, kind in [
        ("Logistic Regression", "logistic"),
        ("Random Forest", "random_forest"),
        ("LightGBM", "lightgbm"),
    ]:
        model = make_pipeline(kind)
        model.fit(x_train, y_train)
        models[name] = model
        print(f"Fitted {name} on {len(x_train):,} Dataset A rows", flush=True)

    evaluation = {
        "evaluation_design": "Dataset A stratified 60/20/20 split; Dataset B was not used to fit models, select parameters, or choose the Fatal threshold.",
        "fatal_threshold": FATAL_THRESHOLD,
        "entropy_alpha": ALPHA,
        "seed": SEED,
        "validation": evaluate(y_validation, adaptive_proba(models, x_validation)),
        "dataset_a_test": evaluate(y_test, adaptive_proba(models, x_test)),
        "dataset_b_independent_same_simulator": evaluate(
            dataset_b["severity"].astype(str), adaptive_proba(models, dataset_b[FEATURES])
        ),
    }
    (output_dir / "interactive_model_metrics.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )

    artifact = {
        "models": models,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "labels": LABELS,
        "alpha": ALPHA,
        "beta": 0.0,
        "fatal_threshold": FATAL_THRESHOLD,
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "test_rows": len(x_test),
        "explanation_background": x_train.sample(
            n=min(32, len(x_train)), random_state=SEED
        ).to_dict(orient="records"),
        "training_dataset": "Dataset A, supplied synthetic simulator sample",
        "dataset_b_used_for_fit": False,
        "random_state": SEED,
        "software_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "shap"]
        },
        "note": "Research demonstration only; not suitable for real-world safety decisions.",
    }
    model_path = model_dir / "safer_raae.joblib"
    joblib.dump(artifact, model_path, compress=3)
    print(f"Saved fitted RAAE artifact to {model_path}", flush=True)
    print(json.dumps({
        key: {metric: values[metric] for metric in ["rows", "accuracy", "balanced_accuracy", "macro_f1", "fatal_precision", "fatal_recall", "fatal_f1", "fatal_pr_auc"]}
        for key, values in evaluation.items() if isinstance(values, dict)
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

