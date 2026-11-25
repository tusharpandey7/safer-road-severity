"""Train SAFER, select ensemble/calibration settings on development data, and evaluate once."""
from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_dataset_c import DESTINATION as DATASET_C_PATH
from src.model_service import CATEGORICAL_FEATURES, FEATURES, LABELS, NUMERIC_FEATURES


SEED = 42
ALPHA = 2.0
BETA_GRID = (0.0, 0.5, 1.0, 2.0)
BOOTSTRAP_REPLICATES = 1000


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


def ensemble_proba(model_probs: dict[str, np.ndarray], alpha: float, beta: float) -> np.ndarray:
    """Entropy and optional consensus-disagreement weighted soft vote."""
    names = list(model_probs)
    stack = np.stack([model_probs[name] for name in names], axis=1)
    eps = 1e-12
    entropy = -(stack * np.log(np.clip(stack, eps, 1.0))).sum(axis=2) / np.log(len(LABELS))
    consensus = np.mean(stack, axis=1, keepdims=True)
    disagreement = (
        stack * np.log(np.clip(stack, eps, 1.0) / np.clip(consensus, eps, 1.0))
    ).sum(axis=2) / np.log(len(LABELS))
    logits = -alpha * entropy - beta * disagreement
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights[:, :, None] * stack, axis=1)


def model_configurations(model_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Logistic Regression": model_probs["Logistic Regression"],
        "Random Forest": model_probs["Random Forest"],
        "LightGBM": model_probs["LightGBM"],
        "Uniform soft voting": np.mean(np.stack(list(model_probs.values()), axis=1), axis=1),
        "Entropy adaptive (alpha=2)": ensemble_proba(model_probs, ALPHA, 0.0),
        **{
            f"Entropy + disagreement (beta={beta:g})": ensemble_proba(model_probs, ALPHA, beta)
            for beta in BETA_GRID if beta > 0
        },
    }


def class_argmax(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]


def fatal_threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    result = class_argmax(probabilities)
    result[probabilities[:, LABELS.index("Fatal")] >= threshold] = "Fatal"
    return result


def macro_ece(y: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y)
    per_class = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_index, label in enumerate(LABELS):
        confidence = probabilities[:, class_index]
        truth = (y == label).astype(float)
        ece = 0.0
        for bin_index in range(bins):
            lower, upper = edges[bin_index], edges[bin_index + 1]
            mask = (confidence >= lower) & (confidence <= upper if bin_index == bins - 1 else confidence < upper)
            if mask.any():
                ece += mask.mean() * abs(float(confidence[mask].mean()) - float(truth[mask].mean()))
        per_class.append(ece)
    return float(np.mean(per_class))


def ordered_log_loss(y: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    """Call sklearn log loss with lexicographic labels matching probability columns."""
    ordered = sorted(LABELS)
    column_order = [LABELS.index(label) for label in ordered]
    return float(log_loss(y, probabilities[:, column_order], labels=ordered))


def summarize(y: pd.Series | np.ndarray, probabilities: np.ndarray, predictions: np.ndarray | None = None) -> dict:
    y = np.asarray(y)
    predicted = class_argmax(probabilities) if predictions is None else np.asarray(predictions)
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0)),
        "fatal_precision": float(report["Fatal"]["precision"]),
        "fatal_recall": float(report["Fatal"]["recall"]),
        "fatal_f1": float(report["Fatal"]["f1-score"]),
        "fatal_pr_auc": float(average_precision_score((y == "Fatal").astype(int), probabilities[:, LABELS.index("Fatal")])),
        "multiclass_brier": float(np.mean(np.sum((np.eye(len(LABELS))[np.array([LABELS.index(v) for v in y])] - probabilities) ** 2, axis=1))),
        "log_loss": ordered_log_loss(y, probabilities),
        "macro_ovr_ece": macro_ece(y, probabilities),
        "class_report": report,
        "confusion_matrix": confusion_matrix(y, predicted, labels=LABELS).tolist(),
    }


def fit_calibrator(method: str, probabilities: np.ndarray, y: pd.Series) -> dict:
    if method == "raw":
        return {"method": "raw"}
    models = []
    for index, label in enumerate(LABELS):
        target = (np.asarray(y) == label).astype(int)
        if method == "platt":
            model = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
            model.fit(probabilities[:, index].reshape(-1, 1), target)
            models.append({"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])})
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
            model.fit(probabilities[:, index], target)
            models.append(model)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    return {"method": method, "models": models}


def apply_calibrator(probabilities: np.ndarray, calibrator: dict) -> np.ndarray:
    method = calibrator["method"]
    if method == "raw":
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    columns = []
    for index, model in enumerate(calibrator["models"]):
        if method == "platt":
            column = expit(model["coef"] * probabilities[:, index] + model["intercept"])
        else:
            column = model.predict(probabilities[:, index])
        columns.append(column)
    calibrated = np.column_stack(columns).clip(1e-9, 1.0)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def choose_fatal_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.05, 0.951, 0.01)
    results = []
    for threshold in candidates:
        predictions = fatal_threshold_predictions(probabilities, float(threshold))
        precision = classification_report(y, predictions, labels=LABELS, output_dict=True, zero_division=0)["Fatal"]["precision"]
        f1 = f1_score(np.asarray(y) == "Fatal", predictions == "Fatal", zero_division=0)
        results.append((f1, precision, float(threshold)))
    return max(results, key=lambda row: (row[0], row[1], row[2]))[2]


def paired_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray], split: str) -> list[dict]:
    rng = np.random.default_rng(SEED + len(y))
    indices = rng.integers(0, len(y), size=(BOOTSTRAP_REPLICATES, len(y)))
    true_index = np.asarray([LABELS.index(value) for value in y], dtype=np.int8)
    offsets = (np.arange(BOOTSTRAP_REPLICATES, dtype=np.int64) * (len(LABELS) ** 2))[:, None]

    def bootstrap_scores(prediction: np.ndarray) -> dict[str, np.ndarray]:
        predicted_index = np.asarray([LABELS.index(value) for value in prediction], dtype=np.int8)
        codes = true_index[indices] * len(LABELS) + predicted_index[indices] + offsets
        matrices = np.bincount(codes.ravel(), minlength=BOOTSTRAP_REPLICATES * len(LABELS) ** 2).reshape(BOOTSTRAP_REPLICATES, len(LABELS), len(LABELS))
        true_counts = matrices.sum(axis=2)
        pred_counts = matrices.sum(axis=1)
        true_positive = np.diagonal(matrices, axis1=1, axis2=2)
        recalls = np.divide(true_positive, true_counts, out=np.zeros_like(true_positive, dtype=float), where=true_counts > 0)
        precisions = np.divide(true_positive, pred_counts, out=np.zeros_like(true_positive, dtype=float), where=pred_counts > 0)
        f1s = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(recalls), where=(precisions + recalls) > 0)
        return {
            "accuracy": true_positive.sum(axis=1) / len(y),
            "balanced_accuracy": recalls.mean(axis=1),
            "macro_f1": f1s.mean(axis=1),
            "fatal_recall": recalls[:, LABELS.index("Fatal")],
        }

    uniform = predictions["Uniform soft voting"]
    uniform_scores = bootstrap_scores(uniform)
    rows = []
    for model_name, prediction in predictions.items():
        if model_name == "Uniform soft voting":
            continue
        model_scores = bootstrap_scores(prediction)
        for metric, values in model_scores.items():
            values = values - uniform_scores[metric]
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({
                "Dataset": split,
                "Comparison": f"{model_name} − Uniform soft voting",
                "Metric": metric,
                "Delta": float(np.mean(values)),
                "CI 2.5%": float(lower),
                "CI 97.5%": float(upper),
                "Replicates": BOOTSTRAP_REPLICATES,
            })
    return rows


def reliability_rows(y: pd.Series, probabilities: np.ndarray, split: str, method: str) -> list[dict]:
    rows = []
    y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, 11)
    for index, label in enumerate(LABELS):
        confidence = probabilities[:, index]
        truth = (y == label).astype(int)
        for bin_index in range(10):
            low, high = edges[bin_index], edges[bin_index + 1]
            mask = (confidence >= low) & (confidence <= high if bin_index == 9 else confidence < high)
            if mask.any():
                rows.append({
                    "Dataset": split,
                    "Method": method,
                    "Class": label,
                    "Bin": bin_index + 1,
                    "Mean predicted": float(confidence[mask].mean()),
                    "Observed frequency": float(truth[mask].mean()),
                    "Count": int(mask.sum()),
                })
    return rows


def main() -> None:
    data_dir = ROOT / "data"
    output_dir = ROOT / "outputs"
    model_dir = ROOT / "trained_model"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    dataset_a = pd.read_csv(data_dir / "synthetic_dataset_A_10000.csv")
    dataset_b = pd.read_csv(data_dir / "synthetic_dataset_B_5000.csv")
    dataset_c = pd.read_csv(DATASET_C_PATH)
    x, y = dataset_a[FEATURES].copy(), dataset_a["severity"].astype(str)

    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_development, y_development, test_size=0.25, random_state=SEED + 1, stratify=y_development
    )
    x_calibration, x_validation, y_calibration, y_validation = train_test_split(
        x_holdout, y_holdout, test_size=0.50, random_state=SEED + 2, stratify=y_holdout
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

    frames = {
        "Validation": x_validation,
        "Calibration": x_calibration,
        "Dataset A test": x_test,
        "Dataset B same-simulator": dataset_b[FEATURES],
        "Dataset C prior-shift": dataset_c[FEATURES],
    }
    labels = {
        "Validation": y_validation,
        "Calibration": y_calibration,
        "Dataset A test": y_test,
        "Dataset B same-simulator": dataset_b["severity"].astype(str),
        "Dataset C prior-shift": dataset_c["severity"].astype(str),
    }
    raw_by_split: dict[str, dict[str, np.ndarray]] = {}
    config_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, frame in frames.items():
        per_model = {name: aligned_proba(model, frame) for name, model in models.items()}
        raw_by_split[split] = per_model
        config_by_split[split] = model_configurations(per_model)

    validation_scores = {
        beta: summarize(y_validation, ensemble_proba(raw_by_split["Validation"], ALPHA, beta))["macro_f1"]
        for beta in BETA_GRID
    }
    selected_beta = max(validation_scores, key=lambda beta: (validation_scores[beta], -beta))
    selected_name = f"Entropy + disagreement (beta={selected_beta:g})" if selected_beta else "Entropy adaptive (alpha=2)"
    # Fit each mapping on the calibration partition; select the method on validation only.
    calibration_input = config_by_split["Calibration"][selected_name]
    calibrators = {
        method: fit_calibrator(method, calibration_input, y_calibration)
        for method in ("raw", "platt", "isotonic")
    }
    calibration_validation = {
        method: apply_calibrator(config_by_split["Validation"][selected_name], calibrator)
        for method, calibrator in calibrators.items()
    }
    validation_losses = {
        method: ordered_log_loss(y_validation, probabilities)
        for method, probabilities in calibration_validation.items()
    }
    selected_calibration = min(validation_losses, key=validation_losses.get)
    selected_calibrator = calibrators[selected_calibration]
    selected_threshold = choose_fatal_threshold(y_validation, calibration_validation[selected_calibration])

    metric_rows = []
    prediction_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split in ("Dataset A test", "Dataset B same-simulator", "Dataset C prior-shift"):
        prediction_by_split[split] = {}
        for model_name, probabilities in config_by_split[split].items():
            scores = summarize(labels[split], probabilities)
            metric_rows.append({"Dataset": split, "Model": model_name, **{k: scores[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "fatal_precision", "fatal_recall", "fatal_f1", "fatal_pr_auc", "multiclass_brier", "log_loss", "macro_ovr_ece"]}})
            prediction_by_split[split][model_name] = class_argmax(probabilities)

        calibrated = apply_calibrator(config_by_split[split][selected_name], selected_calibrator)
        calibrated_pred = fatal_threshold_predictions(calibrated, selected_threshold)
        final_scores = summarize(labels[split], calibrated, calibrated_pred)
        metric_rows.append({"Dataset": split, "Model": f"Selected SAFER + {selected_calibration} calibration + Fatal threshold", **{k: final_scores[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "fatal_precision", "fatal_recall", "fatal_f1", "fatal_pr_auc", "multiclass_brier", "log_loss", "macro_ovr_ece"]}})
        prediction_by_split[split]["Selected SAFER (calibrated + tuned threshold)"] = calibrated_pred

    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(output_dir / "ensemble_comparison.csv", index=False)

    bootstrap_rows = []
    for split in ("Dataset A test", "Dataset B same-simulator", "Dataset C prior-shift"):
        bootstrap_rows.extend(paired_bootstrap(np.asarray(labels[split]), prediction_by_split[split], split))
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "paired_bootstrap_comparisons.csv", index=False)

    calibration_metric_rows = []
    reliability = []
    for split in ("Validation", "Dataset A test", "Dataset B same-simulator", "Dataset C prior-shift"):
        raw_probability = config_by_split[split][selected_name]
        for method, calibrator in calibrators.items():
            probability = apply_calibrator(raw_probability, calibrator)
            scores = summarize(labels[split], probability)
            calibration_metric_rows.append({"Dataset": split, "Method": method, "Selected method": method == selected_calibration, "Rows": scores["rows"], "Multiclass Brier": scores["multiclass_brier"], "Log loss": scores["log_loss"], "Macro OVR ECE": scores["macro_ovr_ece"]})
            reliability.extend(reliability_rows(labels[split], probability, split, method))
    pd.DataFrame(calibration_metric_rows).to_csv(output_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(output_dir / "reliability_diagram_points.csv", index=False)

    for split, key in [("Dataset A test", "A"), ("Dataset B same-simulator", "B"), ("Dataset C prior-shift", "C")]:
        calibrated = apply_calibrator(config_by_split[split][selected_name], selected_calibrator)
        predictions = fatal_threshold_predictions(calibrated, selected_threshold)
        report = pd.DataFrame(classification_report(labels[split], predictions, labels=LABELS, output_dict=True, zero_division=0)).T
        report.to_csv(data_dir / f"class_report_{key}_calibrated.csv", index=True, index_label="Severity")
        pd.DataFrame(confusion_matrix(labels[split], predictions, labels=LABELS), index=LABELS, columns=LABELS).to_csv(data_dir / f"confusion_{key}_calibrated.csv", index_label="Actual")

    artifact = {
        "models": models,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "labels": LABELS,
        "alpha": ALPHA,
        "beta": float(selected_beta),
        "calibrator": selected_calibrator,
        "calibration_method": selected_calibration,
        "fatal_threshold": float(selected_threshold),
        "training_rows": len(x_train),
        "calibration_rows": len(x_calibration),
        "validation_rows": len(x_validation),
        "test_rows": len(x_test),
        "explanation_background": x_train.sample(n=min(32, len(x_train)), random_state=SEED).to_dict(orient="records"),
        "training_dataset": "Dataset A; stratified 60% model fit, 10% calibration, 10% validation, 20% final test",
        "dataset_b_used_for_fit_or_selection": False,
        "dataset_c_used_for_fit_or_selection": False,
        "random_state": SEED,
        "software_versions": {package: version(package) for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "shap"]},
        "note": "Synthetic research demonstration only; not suitable for real-world safety decisions.",
    }
    joblib.dump(artifact, model_dir / "safer_raae.joblib", compress=3)

    summary = {
        "evaluation_design": "Dataset A split 60% model fit, 10% calibration, 10% validation, 20% untouched test; Dataset B untouched same-simulator sample; Dataset C is a prior-shift subset of B and is not independent of B.",
        "selection": {
            "alpha": ALPHA,
            "beta_grid": list(BETA_GRID),
            "validation_macro_f1_by_beta": {str(k): v for k, v in validation_scores.items()},
            "selected_beta": float(selected_beta),
            "calibration_validation_log_loss": validation_losses,
            "selected_calibration": selected_calibration,
            "fatal_threshold_selected_on_validation": float(selected_threshold),
        },
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "dataset_c_class_counts": labels["Dataset C prior-shift"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "dataset_c_warning": "C reuses unique rows from B by design; this is a prior-shift stress scenario, not a second independent test.",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "calibration_metrics_path": "outputs/calibration_metrics.csv",
        "comparison_metrics_path": "outputs/ensemble_comparison.csv",
    }
    (output_dir / "interactive_model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Selected configuration: {selected_name}; calibration: {selected_calibration}; Fatal threshold: {selected_threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
