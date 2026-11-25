"""Model comparisons, paired uncertainty, and probability calibration."""
import altair as alt
import pandas as pd
import streamlit as st

from src.model_service import LABELS
from src.ui_helpers import as_percent, load_experiment_table, load_refit_metrics


st.title("Model performance", icon=":material/monitoring:")
st.caption("All displayed scores evaluate synthetic severity labels. These experiments do not establish real-world crash-risk performance.")
st.info(
    "Dataset C is a 2,000-row, low-Fatal-prior stress sample selected from Dataset B without replacement. C is not independent of B and is not a real-world test set.",
    title="Evaluation boundary",
)

comparison = load_experiment_table("ensemble_comparison.csv")
bootstrap = load_experiment_table("paired_bootstrap_comparisons.csv")
calibration = load_experiment_table("calibration_metrics.csv")
reliability = load_experiment_table("reliability_diagram_points.csv")

split_labels = {
    "Dataset A test": "Dataset A · untouched test",
    "Dataset B same-simulator": "Dataset B · same-simulator sample",
    "Dataset C prior-shift": "Dataset C · low-Fatal stress test",
}
selected_label = st.segmented_control(
    "Evaluation sample", list(split_labels.values()), default=split_labels["Dataset A test"], key="experiment_split"
)
split = next(name for name, label in split_labels.items() if label == selected_label)
table = comparison.loc[comparison["Dataset"].eq(split)].copy()
final = table.loc[table["Model"].str.startswith("Selected SAFER")]
if not final.empty:
    final_row = final.iloc[0]
    with st.container(horizontal=True):
        st.metric("Final model accuracy", as_percent(final_row["accuracy"]), border=True)
        st.metric("Balanced accuracy", as_percent(final_row["balanced_accuracy"]), border=True)
        st.metric("Macro-F1", as_percent(final_row["macro_f1"]), border=True)
        st.metric("Fatal precision / recall", f"{as_percent(final_row['fatal_precision'])} / {as_percent(final_row['fatal_recall'])}", border=True)
    st.caption(f"{final_row['Model']}. The selected calibration and Fatal threshold were chosen on Dataset A development data only.")

metric_labels = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
    "fatal_precision": "Fatal precision",
    "fatal_recall": "Fatal recall",
    "fatal_f1": "Fatal F1",
    "fatal_pr_auc": "Fatal PR-AUC",
}
chosen_metric = st.selectbox("Compare configurations by", list(metric_labels), index=2, format_func=lambda v: metric_labels[v])
plot_data = table[["Model", chosen_metric]].rename(columns={chosen_metric: "Score"}).copy()
plot_data["Score"] *= 100
bars = (
    alt.Chart(plot_data).mark_bar(cornerRadiusEnd=4, color="#147D78")
    .encode(
        x=alt.X("Score:Q", title=f"{metric_labels[chosen_metric]} (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Model:N", sort="-x", title=None),
        tooltip=["Model:N", alt.Tooltip("Score:Q", format=".2f")],
    ).properties(height=max(260, 28 * len(plot_data)))
)
st.altair_chart(bars, alt=f"{metric_labels[chosen_metric]} for each SAFER model configuration on {split_labels[split]}")

display = table[["Model", "accuracy", "balanced_accuracy", "macro_f1", "fatal_precision", "fatal_recall", "fatal_f1", "fatal_pr_auc"]].copy()
for column in display.columns[1:]:
    display[column] = display[column].map(lambda value: as_percent(value, 2))
st.dataframe(display, hide_index=True, alt=f"Held-out classification results for all configurations on {split_labels[split]}")

st.subheader("Paired comparison with uniform soft voting")
paired = bootstrap.loc[bootstrap["Dataset"].eq(split)].copy()
configurations = paired["Comparison"].drop_duplicates().tolist()
selected_beta = float(load_refit_metrics()["selection"]["selected_beta"])
adaptive_name = "Entropy adaptive (alpha=2)" if selected_beta == 0 else f"Entropy + disagreement (beta={selected_beta:g})"
preferred = next((name for name in configurations if name.startswith(adaptive_name)), configurations[0])
chosen_comparison = st.selectbox("Configuration difference", configurations, index=configurations.index(preferred))
delta = paired.loc[paired["Comparison"].eq(chosen_comparison)].copy()
delta["Metric"] = delta["Metric"].map({"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1", "fatal_recall": "Fatal recall"})
delta = delta.rename(columns={"Delta": "Mean difference", "CI 2.5%": "95% CI lower", "CI 97.5%": "95% CI upper"})
for column in ["Mean difference", "95% CI lower", "95% CI upper"]:
    delta[column] = delta[column].map(lambda value: as_percent(value, 2))
st.dataframe(delta[["Metric", "Mean difference", "95% CI lower", "95% CI upper", "Replicates"]], hide_index=True, alt="Paired bootstrap confidence intervals for model differences versus uniform soft voting")
st.caption("Intervals resample the same test rows for both models (1,000 paired bootstrap replicates). An interval spanning zero does not support a clear improvement on that sample. Dataset C has only 30 Fatal records, so its Fatal estimates are especially uncertain.")

st.subheader("Calibration and reliability")
cal_split = st.selectbox("Calibration evaluation sample", list(split_labels), index=list(split_labels).index(split), format_func=lambda value: split_labels[value], key="calibration_split")
cal_table = calibration.loc[calibration["Dataset"].eq(cal_split)].copy()
cal_display = cal_table.rename(columns={"Method": "Method", "Rows": "Rows", "Multiclass Brier": "Multiclass Brier", "Log loss": "Log loss", "Macro OVR ECE": "Macro OVR ECE"})
for column in ["Multiclass Brier", "Log loss", "Macro OVR ECE"]:
    cal_display[column] = cal_display[column].map(lambda value: f"{value:.4f}")
st.dataframe(cal_display[["Method", "Selected method", "Rows", "Multiclass Brier", "Log loss", "Macro OVR ECE"]], hide_index=True, alt="Raw, Platt-scaled, and isotonic calibration metrics")
selected_method = str(cal_table.loc[cal_table["Selected method"], "Method"].iloc[0])
cal_class = st.selectbox("Reliability curve class", LABELS, index=2, key="reliability_class")
curve_data = reliability.loc[
    reliability["Dataset"].eq(cal_split)
    & reliability["Class"].eq(cal_class)
    & reliability["Method"].isin(["raw", selected_method])
].copy()
reference = pd.DataFrame({"Mean predicted": [0.0, 1.0], "Observed frequency": [0.0, 1.0]})
reference_line = alt.Chart(reference).mark_line(strokeDash=[5, 4], color="#777").encode(x="Mean predicted:Q", y="Observed frequency:Q")
curves = alt.Chart(curve_data).mark_line(point=True).encode(
    x=alt.X("Mean predicted:Q", title="Mean predicted class probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("Observed frequency:Q", title="Observed class frequency", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Method:N", title="Calibration"),
    tooltip=["Method:N", "Class:N", alt.Tooltip("Mean predicted:Q", format=".1%"), alt.Tooltip("Observed frequency:Q", format=".1%"), alt.Tooltip("Count:Q", format=",")],
)
st.altair_chart(alt.layer(reference_line, curves).properties(height=300), alt=f"Raw and selected calibrated reliability curves for {cal_class} on {split_labels[cal_split]}")
st.caption("Platt and isotonic mappings are fitted on a dedicated Dataset A calibration partition; the calibration method is selected using validation log loss. Calibration is specific to these synthetic labels. Brier, log loss, and ECE are evaluation summaries, not guarantees of real-world probability accuracy.")

with st.expander("Original supplied workbook benchmark"):
    st.caption("These are the earlier supplied benchmark numbers, retained for traceability. The experiment above is newly reproduced by `src/train_model.py` with calibration and paired comparisons.")
    legacy = pd.read_csv("data/metrics_A_test.csv")
    st.dataframe(legacy, hide_index=True, alt="Original supplied model benchmark metrics")
