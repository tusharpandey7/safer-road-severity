"""Model, data-generation notes, provenance, and limitations."""
import pandas as pd
import streamlit as st

from src.model_service import FEATURES, NUMERIC_FEATURES
from src.ui_helpers import as_percent, load_refit_metrics

st.title("Method and limits", icon=":material/menu_book:")
st.caption("A transparent account of the generated data, adaptive model, and what the scores mean.")
st.warning(
    "SAFER is a college research prototype trained on synthetic labels. It must not be used to make real-world road-safety, emergency-response, insurance, or enforcement decisions.",
    title="Research use only",
)

st.subheader("Study design")
st.mermaid_chart(
    """flowchart LR
      A[Dataset A\n10,000 generated records] --> B[Preprocessing\nscale numeric, one-hot categories]
      B --> C1[Class-weighted\nLogistic Regression]
      B --> C2[Class-weighted\nRandom Forest]
      B --> C3[Class-weighted\nLightGBM]
      C1 --> D[Entropy-adaptive\nsoft vote]
      C2 --> D
      C3 --> D
      D --> E[Three-class score\n+ Fatal threshold]
      E --> F[Prediction + local SHAP]
      G[Dataset B\n5,000 separate generated records] --> H[Independent-sample evaluation only]
      D -. evaluated on .-> H
    """,
    alt="SAFER training and evaluation workflow, showing Dataset B held out for evaluation",
)

model_col, data_col = st.columns(2, gap="large")
with model_col:
    with st.container(border=True):
        st.subheader("Model decision")
        st.markdown("The fitted ensemble contains class-weighted Logistic Regression, Random Forest, and LightGBM. Each model returns a three-class score.")
        st.latex(r"H_m(x) = -\frac{\sum_k p_{mk}(x)\ln p_{mk}(x)}{\ln K}")
        st.latex(r"w_m(x) = \frac{\exp(-\alpha H_m(x))}{\sum_j\exp(-\alpha H_j(x))},\quad \alpha=2")
        st.latex(r"P_{SAFER}(k\mid x)=\sum_m w_m(x)p_{mk}(x)")
        st.markdown("The documented Fatal decision cutoff is **0.470**. When the Fatal score is below the cutoff, the model chooses the higher-scoring Slight or Serious class. The supplied final setting has **beta = 0**, so disagreement does not alter the entropy-based weights.")

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
    "- **Dataset A:** 10,000 synthetic rows used for model development and the supplied held-out evaluation.\n"
    "- **Dataset B:** 5,000 independently generated rows, kept separate from model fitting and threshold selection in the documented experiment. It comes from the same generator, so it is not external real-world validation.\n"
    "- **Reported benchmark:** the app displays the supplied metrics workbook and model-card figures. The interactive model artifact in this project is separately refitted from Dataset A using the documented model family and fixed alpha/Fatal cutoff.\n"
    "- **Target scope:** severity conditional on a generated collision record, not probability of a collision occurring."
)

refit = load_refit_metrics()
st.subheader("Metrics for the model artifact included with this app")
app_test = refit["dataset_a_test"]
app_b = refit["dataset_b_independent_same_simulator"]
with st.container(horizontal=True):
    st.metric("Dataset A held-out accuracy", as_percent(app_test["accuracy"]), border=True)
    st.metric("Dataset A held-out macro-F1", as_percent(app_test["macro_f1"]), border=True)
    st.metric("Dataset B same-simulator accuracy", as_percent(app_b["accuracy"]), border=True)
    st.metric("Dataset B Fatal recall", as_percent(app_b["fatal_recall"]), border=True)
st.caption("These results were computed by the included refit script. They are close to, but not identical with, the supplied report's metrics because the original estimator settings and exact split indices were not included among the supplied files.")

