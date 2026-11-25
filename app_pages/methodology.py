"""Model, data-generation notes, provenance, and limitations."""
import pandas as pd
import streamlit as st

from src.model_service import FEATURES, NUMERIC_FEATURES
from src.ui_helpers import as_percent, load_experiment_table, load_refit_metrics

st.title("Method and limits", icon=":material/menu_book:")
st.caption("A transparent account of the generated data, adaptive model, and what the scores mean.")
st.warning(
    "SAFER is a college research prototype trained on synthetic labels. It must not be used to make real-world road-safety, emergency-response, insurance, or enforcement decisions.",
    title="Research use only",
)

st.subheader("Study design")
st.mermaid_chart(
    """flowchart LR
      A[Dataset A\n10,000 generated records] --> B[60% train\n10% calibrate\n10% validate\n20% final test]
      B --> C1[Class-weighted\nLogistic Regression]
      B --> C2[Class-weighted\nRandom Forest]
      B --> C3[Class-weighted\nLightGBM]
      C1 --> D[Compare uniform, entropy,\nand entropy+disagreement]
      C2 --> D
      C3 --> D
      D --> E[Validation-selected\ncalibration + Fatal cutoff]
      E --> F[Prediction + local SHAP]
      G[Dataset B\n5,000 separate generated records] --> H[Independent-sample evaluation only]
      D -. evaluated on .-> H
      I[Dataset C\n2,000-row low-Fatal prior shift from B] --> J[Stress test only; overlaps B]
      E -. evaluated on .-> J
    """,
    alt="SAFER training and evaluation workflow, showing Dataset B held out for evaluation",
)

model_col, data_col = st.columns(2, gap="large")
with model_col:
    with st.container(border=True):
        st.subheader("Model decision")
        st.markdown("The fitted ensemble contains class-weighted Logistic Regression, Random Forest, and LightGBM. Each model returns a three-class score.")
        st.latex(r"H_m(x) = -\frac{\sum_k p_{mk}(x)\ln p_{mk}(x)}{\ln K}")
        st.latex(r"D_m(x)=\frac{D_{KL}(p_m\Vert \bar p)}{\ln K},\quad w_m(x)=\frac{\exp(-\alpha H_m(x)-\beta D_m(x))}{\sum_j\exp(-\alpha H_j(x)-\beta D_j(x))}")
        st.latex(r"P_{SAFER}(k\mid x)=\sum_m w_m(x)p_{mk}(x)")
        st.markdown("The alpha/beta ensemble configuration and calibration method are selected using Dataset A development partitions only. The Fatal decision cutoff is selected on validation data after calibration. The final beta may be zero; the comparison table reports whether disagreement helped, rather than forcing it into the final configuration.")

with data_col:
    with st.container(border=True):
        st.subheader("What generated the target")
        st.markdown("The severity label comes from a declared risk simulator. It combines weighted context mappings, three interactions, and small Gaussian noise. The coefficients are study assumptions, not fitted physical laws or causal effects.")
        st.latex(r"R=\operatorname{clip}(R_{base}+0.10(r_{speed}r_{vehicle})+0.08(r_{weather}r_{surface})+0.06(r_{light}r_{hour})+\epsilon, 0, 1.35)")
        st.markdown("Thresholds: **Slight** below 0.299; **Serious** from 0.299 to below 0.393; **Fatal** at or above 0.393.")

st.subheader("Model inputs")
definitions = {
    "day_type": "Weekday or weekend context",
    "hour": "Hour of day on a 24-hour clock",
    "speed_limit_kmh": "Posted speed limit in km/h",
    "road_type": "Road category",
    "junction_detail": "Junction configuration label",
    "junction_control": "Reported junction-control category",
    "urban_or_rural": "Urban or rural label",
    "weather_conditions": "Synthetic weather category",
    "light_conditions": "Lighting category",
    "road_surface_conditions": "Surface category",
    "vehicle_type_involved": "Vehicle category",
    "number_of_vehicles": "Number of vehicles in the scenario",
    "driver_age_band": "Driver age group",
}
features = pd.DataFrame({
    "Input": [name.replace("_", " ").capitalize() for name in FEATURES],
    "Type": ["Numeric" if name in NUMERIC_FEATURES else "Categorical" for name in FEATURES],
    "Description": [definitions[name] for name in FEATURES],
})
st.dataframe(features, hide_index=True, alt="Thirteen model inputs and plain-language definitions")
st.caption("collision_date is present in the data but intentionally excluded from the predictor. This package does not call a live weather API and does not contain vehicle-history, traffic-flow, speed-at-impact, or road-geometry measurements.")

st.subheader("Data and evaluation boundary")
st.markdown(
    "- **Dataset A:** 10,000 synthetic records. A stratified split reserves 60% for fitting, 10% for calibration, 10% for model/calibration/threshold selection, and 20% as the final test.\n"
    "- **Dataset B:** 5,000 separately generated records from the same simulator. It is not used for fitting or selection and is not real-world external validation.\n"
    "- **Dataset C:** 2,000 unique rows sampled without replacement from B: 75% Slight, 23.5% Serious, 1.5% Fatal. C preserves the source feature/label pairs and changes the class prior only. Since C is a subset of B, their results are not independent replications. The target prior is a STATS19-inspired stress scenario, not a measured prevalence claim in this synthetic file.\n"
    "- **Calibration:** one-vs-rest Platt scaling and isotonic mappings are fitted on the calibration partition; raw, Platt, and isotonic probabilities are compared on validation data. The chosen method and Fatal cutoff are then evaluated on untouched A test data, B, and C.\n"
    "- **Comparison:** all base learners and voting configurations use the same test rows. Paired bootstrap intervals compare each configuration with uniform soft voting.\n"
    "- **Target scope:** severity conditional on a generated collision record, not probability of a collision occurring. No result here establishes real-road performance."
)

refit = load_refit_metrics()
st.subheader("Metrics for the model artifact included with this app")
all_results = load_experiment_table("ensemble_comparison.csv")
selected = all_results.loc[all_results["Model"].str.startswith("Selected SAFER")].set_index("Dataset")
app_test = selected.loc["Dataset A test"]
app_b = selected.loc["Dataset B same-simulator"]
app_c = selected.loc["Dataset C prior-shift"]
with st.container(horizontal=True):
    st.metric("Dataset A held-out accuracy", as_percent(app_test["accuracy"]), border=True)
    st.metric("Dataset A held-out macro-F1", as_percent(app_test["macro_f1"]), border=True)
    st.metric("Dataset B Fatal recall", as_percent(app_b["fatal_recall"]), border=True)
    st.metric("Dataset C Fatal recall", as_percent(app_c["fatal_recall"]), border=True)
st.caption(f"Selected beta={refit['selection']['selected_beta']:g}; calibration={refit['selection']['selected_calibration']}; validation-selected Fatal alert cutoff={refit['selection']['fatal_threshold_selected_on_validation']:.2f}. C is a prior-shift subset of B, not independent evidence. These results are synthetic-simulator results; calibrated scores should not be interpreted as real-world risk probabilities.")

