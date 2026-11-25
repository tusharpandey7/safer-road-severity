"""Benchmark, class-level metrics, and error analysis."""
import altair as alt
import pandas as pd
import streamlit as st

from src.model_service import LABELS
from src.ui_helpers import (as_percent, load_ablation_table, load_matrix,
                            load_metric_table, load_refit_metrics, load_report)


st.title("Model performance", icon=":material/monitoring:")
st.caption("Metrics below are copied from the supplied SAFER workbook and model card; both evaluation sets have synthetic labels.")
st.info(
    "Dataset B is a separate sample from the same simulator. It checks repeatability across generated records, not transfer to real roads or a different population.",
    title="Read the benchmark carefully",
)

split = st.segmented_control("Evaluation set", ["Dataset A test", "Dataset B independent"], default="Dataset A test", key="metric_split")
which = "A" if split == "Dataset A test" else "B"
table = load_metric_table(which)
primary = table.loc[table["Model"] == "SAFER RAAE"].iloc[0]
report = load_report(which)
cm = load_matrix(which)

with st.container(horizontal=True):
    st.metric("Accuracy", as_percent(primary["Accuracy"]), border=True)
    st.metric("Balanced accuracy", as_percent(primary["Balanced Accuracy"]), border=True)
    st.metric("Macro-F1", as_percent(primary["Macro F1"]), border=True)
    st.metric("Fatal precision / recall", f"{as_percent(primary['Fatal Precision'])} / {as_percent(primary['Fatal Recall'])}", border=True)

metric_choice = st.selectbox(
    "Compare models by metric",
    ["Accuracy", "Balanced Accuracy", "Macro F1", "Fatal Precision", "Fatal Recall", "Fatal PR-AUC"],
    index=2,
)
comparison = table[["Model", metric_choice]].copy()
comparison["Score"] = comparison[metric_choice] * 100
bars = (
    alt.Chart(comparison)
    .mark_bar(cornerRadiusEnd=4, color="#147D78")
    .encode(
        x=alt.X("Score:Q", title=f"{metric_choice} (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Model:N", sort="-x", title=None),
        tooltip=["Model:N", alt.Tooltip("Score:Q", format=".2f")],
    )
    .properties(height=260)
)
st.altair_chart(bars, alt=f"{metric_choice} model comparison for {split}")

left, right = st.columns([0.88, 1.12], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Class-level results")
        class_rows = report[report.iloc[:, 0].isin(LABELS)].copy()
        class_rows = class_rows.rename(columns={class_rows.columns[0]: "Severity", "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"})
        for metric in ["Precision", "Recall", "F1"]:
            class_rows[metric] = class_rows[metric].map(lambda v: as_percent(v, 1))
        st.dataframe(class_rows, hide_index=True, alt=f"Class-level precision, recall, F1, and support for {split}")
        st.caption("Support is the number of records in each synthetic class.")

with right:
    with st.container(border=True):
        st.subheader("Confusion matrix")
        cm_long = cm.rename_axis("Actual").reset_index().melt(id_vars="Actual", var_name="Predicted", value_name="Records")
        heat = (
            alt.Chart(cm_long)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Predicted:N", sort=LABELS, title="Predicted class"),
                y=alt.Y("Actual:N", sort=LABELS, title="Actual class"),
                color=alt.Color("Records:Q", scale=alt.Scale(scheme="tealblues"), title="Records"),
                tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Records:Q", format=",")],
            )
            .properties(height=260)
        )
        st.altair_chart(heat, alt=f"Confusion matrix for {split}; rows are actual and columns predicted")

with st.expander("Inspect the supplied ablation study", icon=":material/science:"):
    ablation = load_ablation_table().copy()
    selected = ablation[["Model", "Accuracy", "Balanced Accuracy", "Macro F1", "Fatal Precision", "Fatal Recall", "Fatal F1"]].copy()
    for column in selected.columns[1:]:
        selected[column] = selected[column].map(lambda value: as_percent(value, 1))
    st.dataframe(selected, hide_index=True, alt="Ablation study of uniform voting, entropy adaptation, disagreement, and Fatal threshold")
    st.caption("The final configuration has beta = 0, so disagreement does not change the entropy-only adaptive weights. SAFER RAAE and entropy-only adaptive therefore have the same recorded scores.")

st.caption("The class scores are evaluated on generated labels. These reported metrics do not establish calibration or real-world safety performance.")

st.subheader("Interactive model artifact from this project")
st.caption("These are separately recomputed after fitting the included training script. Dataset B remains outside fitting, preprocessing, and threshold selection.")
refit = load_refit_metrics()
refit_key = "dataset_a_test" if which == "A" else "dataset_b_independent_same_simulator"
measured = refit[refit_key]
with st.container(horizontal=True):
    st.metric("Refit accuracy", as_percent(measured["accuracy"]), border=True)
    st.metric("Refit balanced accuracy", as_percent(measured["balanced_accuracy"]), border=True)
    st.metric("Refit macro-F1", as_percent(measured["macro_f1"]), border=True)
    st.metric("Refit Fatal precision / recall", f"{as_percent(measured['fatal_precision'])} / {as_percent(measured['fatal_recall'])}", border=True)
st.caption(f"Evaluation rows: {measured['rows']:,}. Split: {refit['evaluation_design']} The displayed model-card metrics above remain the numbers copied from the supplied workbook.")
