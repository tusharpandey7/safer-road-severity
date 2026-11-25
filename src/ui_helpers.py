"""Shared cached loaders and formatting helpers for the Streamlit UI."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.data_service import read_ablation, read_class_report, read_confusion, read_dataset, read_metrics
from src.model_service import load_predictor
from src.data_service import ROOT


@st.cache_data
def load_data(which: str) -> pd.DataFrame:
    return read_dataset(which)


@st.cache_data
def load_metric_table(which: str) -> pd.DataFrame:
    return read_metrics(which)


@st.cache_data
def load_report(which: str) -> pd.DataFrame:
    return read_class_report(which)


@st.cache_data
def load_matrix(which: str) -> pd.DataFrame:
    return read_confusion(which)


@st.cache_data
def load_ablation_table() -> pd.DataFrame:
    return read_ablation()


@st.cache_data
def load_refit_metrics() -> dict:
    path = ROOT / "outputs" / "interactive_model_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return load_predictor()


def as_percent(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}%}"
