"""Small EDA and downloadable view of the supplied synthetic datasets."""
import altair as alt
import pandas as pd
import streamlit as st

from src.model_service import FEATURES, LABELS
from src.ui_helpers import load_data

st.title("Dataset explorer", icon=":material/table_chart:")
st.caption("Inspect the two simulator-generated tables supplied with the SAFER package.")
st.warning("These are synthetic records. The weather, road context, collision date, and severity labels are simulated; this page is not a live government data feed.", title="Synthetic data")

choice = st.segmented_control("Dataset", ["Dataset A · development", "Dataset B · independent sample"], default="Dataset A · development", key="explorer_set")
which = "A" if choice == "Dataset A · development" else "B"
frame = load_data(which)
target_counts = frame["severity"].value_counts().reindex(LABELS, fill_value=0)
sample_period = f"{frame['collision_date'].min():%Y-%m-%d} to {frame['collision_date'].max():%Y-%m-%d}"

with st.container(horizontal=True):
    st.metric("Rows", f"{len(frame):,}", border=True)
    st.metric("Input fields", len(FEATURES), border=True)
    st.metric("Synthetic target classes", len(LABELS), border=True)
    st.metric("Generated date span", sample_period, border=True)

left, right = st.columns([0.85, 1.15], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Severity label distribution")
        counts = target_counts.rename_axis("Severity").reset_index(name="Records")
        chart = (
            alt.Chart(counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Severity:N", sort=LABELS, title=None),
                y=alt.Y("Records:Q", title="Synthetic records"),
                color=alt.Color("Severity:N", scale=alt.Scale(domain=LABELS, range=["#138A80", "#D69E2E", "#D9534F"]), legend=None),
                tooltip=["Severity:N", alt.Tooltip("Records:Q", format=",")],
            ).properties(height=250)
        )
        st.altair_chart(chart, alt=f"Synthetic class counts in dataset {which}")

with right:
    with st.container(border=True):
        st.subheader("How severity varies across one field")
        factor_options = [
            "weather_conditions", "light_conditions", "road_surface_conditions",
            "road_type", "urban_or_rural", "vehicle_type_involved", "speed_limit_kmh",
        ]
        factor = st.selectbox("Compare by", factor_options, format_func=lambda value: value.replace("_", " ").capitalize())
        grouped = frame.groupby([factor, "severity"], observed=False).size().rename("records").reset_index()
        grouped["share"] = grouped["records"] / grouped.groupby(factor)["records"].transform("sum")
        mix_chart = (
            alt.Chart(grouped).mark_bar()
            .encode(
                x=alt.X(f"{factor}:N", title=factor.replace("_", " ").capitalize(), axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share within category"),
                color=alt.Color("severity:N", sort=LABELS, scale=alt.Scale(domain=LABELS, range=["#138A80", "#D69E2E", "#D9534F"]), title="Severity"),
                tooltip=[alt.Tooltip(f"{factor}:N", title=factor.replace("_", " ").capitalize()), "severity:N", alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("records:Q", format=",")],
            ).properties(height=250)
        )
        st.altair_chart(mix_chart, alt=f"Synthetic severity composition by {factor.replace('_', ' ')}")
        st.caption("Within-category shares describe the simulator sample. They are not an estimate of real-world risk.")

with st.container(border=True):
    st.subheader("Sample records")
    st.caption("The date is retained for data exploration, but the supplied model card excludes collision_date from predictors.")
    st.dataframe(frame.head(100), hide_index=True, alt=f"First 100 synthetic records from Dataset {which}")
    st.download_button(
        f"Download Dataset {which} CSV", data=frame.to_csv(index=False),
        file_name=f"SAFER_synthetic_dataset_{which}.csv", mime="text/csv",
        icon=":material/download:",
    )
