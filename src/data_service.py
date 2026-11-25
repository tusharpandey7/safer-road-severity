"""Cached paths and metadata for the local SAFER data package."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_A = ROOT / "data" / "synthetic_dataset_A_10000.csv"
DATA_B = ROOT / "data" / "synthetic_dataset_B_5000.csv"
DATA_C = ROOT / "data" / "synthetic_dataset_C_prior_shift_2000.csv"
METRICS_A = ROOT / "data" / "metrics_A_test.csv"
METRICS_B = ROOT / "data" / "metrics_B_independent.csv"
ABLATION_A = ROOT / "data" / "metrics_A_ablation.csv"
REPORT_A = ROOT / "data" / "class_report_A.csv"
REPORT_B = ROOT / "data" / "class_report_B.csv"
CM_A = ROOT / "data" / "confusion_A.csv"
CM_B = ROOT / "data" / "confusion_B.csv"
def read_dataset(which: str) -> pd.DataFrame:
    paths = {"A": DATA_A, "B": DATA_B, "C": DATA_C}
    path = paths[which]
    return pd.read_csv(path, parse_dates=["collision_date"])


def read_metrics(which: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_A if which == "A" else METRICS_B)


def read_class_report(which: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_A if which == "A" else REPORT_B)


def read_confusion(which: str) -> pd.DataFrame:
    return pd.read_csv(CM_A if which == "A" else CM_B, index_col=0)


def read_ablation() -> pd.DataFrame:
    return pd.read_csv(ABLATION_A)


def read_output_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / filename)

