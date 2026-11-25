"""Overview page."""
import altair as alt
import pandas as pd
import streamlit as st

from src.model_service import LABELS
from src.ui_helpers import as_percent, load_data, load_experiment_table


st.title("Road severity, made easier to inspect", icon=":material/alt_route:")
st.caption("SAFER estimates Slight, Serious, or Fatal severity conditional on a hypothetical collision, using pre-crash context fields.")
st.warning(
    "Research prototype: both datasets and their severity labels are synthetic. Scores shown here describe this simulator study, not real-world road risk.",
    title="Know what this model can tell you",
)

experiments = load_experiment_table("ensemble_comparison.csv")
raae_a = experiments.loc[experiments["Dataset"].eq("Dataset A test") & experiments["Model"].str.startswith("Selected SAFER")].iloc[0]
raae_b = experiments.loc[experiments["Dataset"].eq("Dataset B same-simulator") & experiments["Model"].str.startswith("Selected SAFER")].iloc[0]
data_a = load_data("A")
data_b = load_data("B")
data_c = load_data("C")
fatal_share_c = data_c["severity"].eq("Fatal").mean()

with st.container(horizontal=True):
    st.metric("Dataset A test accuracy", as_percent(raae_a["accuracy"]), border=True)
    st.metric("Dataset B accuracy", as_percent(raae_b["accuracy"]), border=True)
    st.metric("Dataset B Fatal recall", as_percent(raae_b["fatal_recall"]), border=True)
    st.metric("Dataset C stress sample", f"{len(data_c):,} rows", border=True)

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    with st.container(border=True):
        st.subheader("A balanced simulator sample")
        counts = (
            data_a["severity"].value_counts()
            .reindex(LABELS, fill_value=0)
            .rename_axis("Severity")
            .reset_index(name="Records")
        )
        chart = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Severity:N", sort=LABELS, title=None),
                y=alt.Y("Records:Q", title="Synthetic records"),
                color=alt.Color(
                    "Severity:N",
                    scale=alt.Scale(domain=LABELS, range=["#138A80", "#D69E2E", "#D9534F"]),
                    legend=None,
                ),
                tooltip=["Severity:N", alt.Tooltip("Records:Q", format=",")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, alt="Dataset A class counts: slight, serious, and fatal synthetic labels")
        st.caption("Dataset A was generated with fixed simulator thresholds; its class balance is not a measure of real crash prevalence.")
        st.caption(f"Dataset C uses a {fatal_share_c:.2%} synthetic Fatal class prior and contains only {int((data_c['severity'] == 'Fatal').sum())} Fatal rows. It is a no-replacement subset of B, so it is not independent of B.")

with right:
    with st.container(border=True):
        st.subheader("How to use the app")
        st.markdown(
            "1. Open **Scenario lab**.\n"
            "2. Enter the 13 fields available in the SAFER dataset.\n"
            "3. Review the class scores and the local SHAP explanation.\n"
            "4. Open **Model performance** to compare model configurations, paired uncertainty intervals, and calibration."
        )
        st.info(
            "The displayed class scores are calibrated against synthetic validation data. They are not probabilities of a real-world crash outcome.",
            title="Interpretation",
        )

st.subheader("What SAFER is designed to do")
with st.container(horizontal=True):
    with st.container(border=True, height="stretch"):
        st.markdown("**Estimate severity before impact**")
        st.caption("The target is severity conditional on a hypothetical collision—not the chance that one will happen.")
    with st.container(border=True, height="stretch"):
        st.markdown("**Show its reasoning**")
        st.caption("Permutation SHAP attributes the adaptive ensemble's class score across the entered fields.")
    with st.container(border=True, height="stretch"):
        st.markdown("**Keep its limits visible**")
        st.caption("Dataset B is an independent synthetic draw, not external validation on real crashes. Simulated speed is not measured speed.")
