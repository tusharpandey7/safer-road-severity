"""Generate pre-crash-only synthetic scenarios A and B from a frozen specification.

The category marginals are a frozen description of the earlier synthetic sample.
No crash-count/vehicle-count variable is generated. Pre-impact speed is explicitly
simulated (not observed), and speed excess / speeding are derived from it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MARGINALS_PATH = DATA_DIR / "synthetic_context_marginals.json"
MAPPINGS_PATH = DATA_DIR / "risk_mappings.csv"
DATASETS = {
    "A": (10_000, 42, DATA_DIR / "synthetic_dataset_A_10000.csv"),
    "B": (5_000, 2026, DATA_DIR / "synthetic_dataset_B_5000.csv"),
}
LABELS = ("Slight", "Serious", "Fatal")
THRESHOLDS = (0.299, 0.393)
WEIGHTS = {
    "speed_limit": 0.18,
    "vehicle_type": 0.16,
    "road_type": 0.08,
    "junction_detail": 0.08,
    "junction_control": 0.04,
    "urban_or_rural": 0.04,
    "weather": 0.08,
    "light": 0.08,
    "surface": 0.08,
    "driver_age": 0.06,
    "hour": 0.04,
    "speed_excess": 0.08,
}
INTERACTIONS = {"speed_excess_x_vehicle": 0.10, "weather_x_surface": 0.08, "light_x_hour": 0.06}
NOISE_SD = 0.018
SPEED_DELTA_VALUES = np.array([-20, -10, 0, 5, 10, 15, 20, 30], dtype=int)
SPEED_DELTA_PROBABILITIES = np.array([0.08, 0.12, 0.25, 0.15, 0.14, 0.12, 0.09, 0.05])


def risk_lookup() -> dict[str, dict[str, float]]:
    frame = pd.read_csv(MAPPINGS_PATH)
    return {
        feature: dict(zip(group["Category/range"].astype(str), group["Normalized risk contribution"].astype(float)))
        for feature, group in frame.groupby("Feature", sort=False)
    }


def _category_risk(values: pd.Series, mapping: dict[str, float]) -> np.ndarray:
    return values.astype(str).map(mapping).astype(float).to_numpy()


def risk_components(frame: pd.DataFrame) -> pd.DataFrame:
    maps = risk_lookup()
    components = pd.DataFrame(index=frame.index)
    speed = pd.to_numeric(frame["speed_limit_kmh"], errors="coerce")
    components["speed_limit"] = np.select([speed <= 30, speed <= 50, speed <= 70, speed <= 90], [0.05, 0.20, 0.55, 0.82], default=1.0)
    mappings = {
        "vehicle_type": ("vehicle_type_involved", "vehicle_type_involved"),
        "road_type": ("road_type", "road_type"),
        "junction_detail": ("junction_detail", "junction_detail"),
        "junction_control": ("junction_control", "junction_control"),
        "urban_or_rural": ("urban_or_rural", "urban_or_rural"),
        "weather": ("weather_conditions", "weather_conditions"),
        "light": ("light_conditions", "light_conditions"),
        "surface": ("road_surface_conditions", "road_surface_conditions"),
        "driver_age": ("driver_age_band", "driver_age_band"),
    }
    for key, (column, mapping_key) in mappings.items():
        components[key] = _category_risk(frame[column], maps[mapping_key])
    hour = pd.to_numeric(frame["hour"], errors="coerce")
    components["hour"] = np.where(hour < 5, 0.70, np.where(hour.between(17, 20), 0.45, 0.20))
    excess = pd.to_numeric(frame["speed_excess_kmh_simulated"], errors="coerce").clip(lower=0)
    components["speed_excess"] = np.minimum(excess / 30.0, 1.0)
    return components


def calculate_risk(frame: pd.DataFrame, noise: np.ndarray | None = None, weights: dict[str, float] | None = None, interactions: dict[str, float] | None = None) -> np.ndarray:
    components = risk_components(frame)
    weights = WEIGHTS if weights is None else weights
    interactions = INTERACTIONS if interactions is None else interactions
    base = sum(float(weights[key]) * components[key].to_numpy() for key in weights)
    interaction = (
        interactions["speed_excess_x_vehicle"] * components["speed_excess"] * components["vehicle_type"]
        + interactions["weather_x_surface"] * components["weather"] * components["surface"]
        + interactions["light_x_hour"] * components["light"] * components["hour"]
    )
    epsilon = np.zeros(len(frame), dtype=float) if noise is None else np.asarray(noise, dtype=float)
    return np.clip(base + interaction + epsilon, 0.0, 1.35)


def severity_from_risk(risk: np.ndarray, thresholds: tuple[float, float] = THRESHOLDS) -> np.ndarray:
    return np.select([risk < thresholds[0], risk < thresholds[1]], ["Slight", "Serious"], default="Fatal")


def generate_dataset(name: str) -> pd.DataFrame:
    rows, seed, destination = DATASETS[name]
    rng = np.random.default_rng(seed)
    marginal_spec = json.loads(MARGINALS_PATH.read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    date_start, date_end = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31")
    day_offsets = rng.integers(0, (date_end - date_start).days + 1, size=rows)
    dates = date_start + pd.to_timedelta(day_offsets, unit="D")
    out["collision_date"] = dates.strftime("%Y-%m-%d").to_numpy()
    out["day_type"] = np.where(dates.day_of_week < 5, "Weekday", "Weekend")
    feature_marginals = marginal_spec["categorical_marginals"]
    for column, counts in feature_marginals.items():
        categories = list(counts)
        probabilities = np.asarray(list(counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        out[column] = rng.choice(categories, size=rows, p=probabilities)
    hours = marginal_spec["hour_marginal"]
    hour_probabilities = np.asarray(list(hours.values()), dtype=float)
    hour_probabilities /= hour_probabilities.sum()
    out["hour"] = rng.choice(np.asarray(list(hours), dtype=int), size=rows, p=hour_probabilities)
    speed_limits = pd.to_numeric(pd.Series(out["speed_limit_kmh"])).astype(int).to_numpy()
    speed_delta = rng.choice(SPEED_DELTA_VALUES, size=rows, p=SPEED_DELTA_PROBABILITIES)
    actual_speed = np.clip(speed_limits + speed_delta, 0, 160).astype(int)
    speed_excess = np.maximum(actual_speed - speed_limits, 0).astype(int)
    out["speed_limit_kmh"] = speed_limits
    out["actual_speed_kmh_simulated"] = actual_speed
    out["speed_excess_kmh_simulated"] = speed_excess
    out["speeding_indicator_simulated"] = (speed_excess > 0).astype(int)
    frame = pd.DataFrame(out)
    noise = rng.normal(0.0, NOISE_SD, size=rows)
    simulator_risk = calculate_risk(frame, noise=noise)
    frame["simulator_noise"] = noise
    frame["simulator_risk_score"] = simulator_risk
    frame["severity"] = severity_from_risk(simulator_risk)
    frame.to_csv(destination, index=False)
    return frame


def main() -> None:
    metadata = {
        "scope": "Synthetic pre-crash severity conditional on a hypothetical collision occurring",
        "feature_availability": "number_of_vehicles is excluded because it is generally only known after a collision",
        "speed_note": "actual_speed_kmh_simulated, speed_excess_kmh_simulated, and speeding_indicator_simulated are hypothetical simulator values, not measured accident data",
        "marginal_source": "Frozen category marginals estimated from the earlier synthetic Dataset A, stored in synthetic_context_marginals.json; not real-world frequencies",
        "weights": WEIGHTS,
        "interactions": INTERACTIONS,
        "noise": f"Normal(0, {NOISE_SD}^2), independently sampled per generated row",
        "thresholds": {"Slight_upper": THRESHOLDS[0], "Serious_upper": THRESHOLDS[1], "Fatal_lower": THRESHOLDS[1]},
        "seed_and_rows": {name: {"rows": rows, "seed": seed} for name, (rows, seed, _) in DATASETS.items()},
        "limitations": ["Weights, speed residual distribution, noise, and severity thresholds are transparent design assumptions, not learned physical laws or causal effects.", "These synthetic labels cannot establish real-world prediction performance."],
    }
    for name in DATASETS:
        frame = generate_dataset(name)
        metadata.setdefault("class_counts", {})[name] = frame["severity"].value_counts().to_dict()
        print(name, len(frame), frame["severity"].value_counts().to_dict(), flush=True)
    (DATA_DIR / "synthetic_generation_specification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
