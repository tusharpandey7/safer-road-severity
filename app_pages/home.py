"""Overview page."""
import altair as alt
import pandas as pd
import streamlit as st

from src.model_service import LABELS
from src.ui_helpers import as_percent, load_data, load_metric_table


st.title("Road severity, made easier to inspect", icon=":material/alt_route:")
st.caption("SAFER turns a set of collision-context fields into a Slight, Serious, or Fatal severity classification.")
st.warning(
    "Research prototype: both datasets and their severity labels are synthetic. Scores shown here describe this simulator study, not real-world road risk.",
    title="Know what this model can tell you",
)

metrics_a = load_metric_table("A")
metrics_b = load_metric_table("B")
raae_a = metrics_a.loc[metrics_a["Model"] == "SAFER RAAE"].iloc[0]
raae_b = metrics_b.loc[metrics_b["Model"] == "SAFER RAAE"].iloc[0]
data_a = load_data("A")
data_b = load_data("B")

with st.container(horizontal=True):
    st.metric("Dataset A test accuracy", as_percent(raae_a["Accuracy"]), border=True)
    st.metric("Dataset B accuracy", as_percent(raae_b["Accuracy"]), border=True)
    st.metric("Dataset B Fatal recall", as_percent(raae_b["Fatal Recall"]), border=True)
    st.metric("Synthetic records", f"{len(data_a) + len(data_b):,}", border=True)

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

with right:
    with st.container(border=True):
        st.subheader("How to use the app")
        st.markdown(
            "1. Open **Scenario lab**.\n"
            "2. Enter the 13 fields available in the SAFER dataset.\n"
            "3. Review the class scores and the local SHAP explanation.\n"
            "4. Open **Model performance** to inspect class-level results and error patterns."
        )
        st.info(
            "The displayed percentages are model scores. They are not calibrated probabilities of a real crash outcome.",
            title="Interpretation",
        )

st.subheader("What SAFER is designed to do")
with st.container(horizontal=True):
    with st.container(border=True, height="stretch"):
        st.markdown("**Classify a recorded scenario**")
        st.caption("The target is severity among generated collision records—not whether a collision will happen.")
    with st.container(border=True, height="stretch"):
        st.markdown("**Show its reasoning**")
        st.caption("Permutation SHAP attributes the adaptive ensemble's class score across the entered fields.")
    with st.container(border=True, height="stretch"):
        st.markdown("**Keep its limits visible**")
        st.caption("Dataset B is an independent draw from the same generator, not external validation on real crashes.")
