"""Study design, simulated target, pre-crash schema, and limits."""
import pandas as pd
import streamlit as st

from src.model_service import FEATURES, NUMERIC_FEATURES
from src.ui_helpers import as_percent, load_experiment_table, load_refit_metrics


st.title("Method and limits", icon=":material/menu_book:")
st.caption("SAFER studies severity classification conditional on a hypothetical collision, using only scenario features available before impact in its intended formulation.")
st.warning(
    "This is a synthetic research prototype. Speed values are simulated, and results do not establish real-world pre-crash performance or safety benefit.",
    title="Research use only",
)

st.subheader("Study design")
st.mermaid_chart(
    """flowchart LR
      A[Dataset A\n10,000 simulated scenarios] --> B[60% fit\n10% calibrate\n10% validate\n20% final test]
      B --> C[Logistic Regression\nRandom Forest\nLightGBM]
      C --> D[Uniform and entropy\nadaptive voting comparison]
      D --> E[Validation-selected\ncalibration and Fatal cutoff]
      E --> F[Class scores\nlocal SHAP\nscenario comparison]
      G[Dataset B\n5,000 independent simulator draws] --> H[Same-simulator evaluation]
      I[Dataset C\n2,000-row low-Fatal prior shift] --> J[Imbalance stress test\nsubset of B]
      E -. evaluated on .-> H
      E -. evaluated on .-> J
    """,
    alt="SAFER uses simulated pre-impact fields, separates model development and evaluation, and reports same-simulator and class-prior stress tests",
)

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.subheader("Simulator equation")
        st.markdown("The risk target is a designed label rule. Its coefficients are transparent experiment settings, not values learned from real crashes or established physical constants.")
        st.latex(r"R_{base}=0.18r_{limit}+0.16r_{vehicle}+0.08r_{road}+0.08r_{junction}+0.04r_{control}+0.04r_{area}+0.08r_{weather}+0.08r_{light}+0.08r_{surface}+0.06r_{age}+0.04r_{hour}+0.08r_{speed\ excess}")
        st.latex(r"I=0.10(r_{speed\ excess}r_{vehicle})+0.08(r_{weather}r_{surface})+0.06(r_{light}r_{hour})")
        st.latex(r"R=\operatorname{clip}(R_{base}+I+\epsilon,0,1.35),\quad\epsilon\sim N(0,0.018^2)")
        st.markdown("Severity: **Slight** for R < 0.299; **Serious** for 0.299 ≤ R < 0.393; **Fatal** for R ≥ 0.393.")
with right:
    with st.container(border=True):
        st.subheader("Why these settings")
        justification = pd.DataFrame([
            ("Posted speed limit", "0.18", "Higher relative weight for road-speed context; it is a legal limit, not the speed driven."),
            ("Vehicle category", "0.16", "Represents designed differences between vehicle categories."),
            ("Road / junction / area", "0.08 / 0.08 / 0.04", "Represents road layout and setting in the controlled scenario."),
            ("Weather / lighting / surface", "0.08 each", "Represents designed environmental and visibility conditions."),
            ("Driver age band / hour", "0.06 / 0.04", "Adds age-group and time-of-day variation to the simulator."),
            ("Speed excess", "0.08", "Substitutes for post-collision vehicle count with a simulated pre-impact signal."),
            ("Speed excess × vehicle", "0.10", "Designed compounding interaction for the simulation."),
            ("Gaussian noise", "σ = 0.018", "Adds controlled variation; it is not measurement error estimated from observed crashes."),
        ], columns=["Parameter", "Setting", "Design rationale"])
        st.dataframe(justification, hide_index=True, alt="Transparent design settings and qualitative reasons; coefficients are not empirically estimated")
        st.caption("These rationales are qualitative road-safety design choices only. No regression estimation, expert elicitation, or causal study established the numerical values.")

st.subheader("Predictor availability audit")
availability = pd.DataFrame([
    ("Posted speed limit", "Known road attribute before impact", "Model input; distinct from driving speed"),
    ("Hypothetical pre-impact speed", "Would require measured/estimated vehicle speed", "Synthetic-only input here; not telematics or police measurement"),
    ("Speed excess and speeding indicator", "Derived from the simulated speed and posted limit", "Engineered predictors in this synthetic experiment"),
    ("Road, junction, area, weather, light, surface", "Potentially observable before impact", "Simulator inputs; no live API or sensor feed"),
    ("Vehicle category, driver age band", "May be known only in some settings", "Assumed available in the scenario form"),
    ("Number of vehicles involved", "Usually known only after a collision", "Removed from the generated data and model"),
    ("Simulator noise, risk score, severity label", "Generated target-construction fields", "Retained for audit; excluded from model predictors"),
    ("Collision date", "Temporal metadata", "Retained in data; excluded from predictor list"),
], columns=["Field", "Availability", "SAFER treatment"])
st.dataframe(availability, hide_index=True, alt="Pre-crash feature availability audit and inclusion decisions")

definitions = {
    "day_type": "Weekday or weekend",
    "hour": "Hour on a 24-hour clock",
    "speed_limit_kmh": "Posted legal limit in km/h",
    "road_type": "Road category",
    "junction_detail": "Junction configuration",
    "junction_control": "Junction-control category",
    "urban_or_rural": "Urban or rural setting",
    "weather_conditions": "Synthetic weather category",
    "light_conditions": "Lighting category",
    "road_surface_conditions": "Synthetic surface category",
    "vehicle_type_involved": "Vehicle category",
    "driver_age_band": "Driver age group",
    "actual_speed_kmh_simulated": "Hypothetical simulator pre-impact speed, not an observation",
    "speed_excess_kmh_simulated": "max(0, simulated speed − posted limit)",
    "speeding_indicator_simulated": "1 when simulated speed exceeds posted limit; otherwise 0",
}
feature_table = pd.DataFrame({
    "Model feature": FEATURES,
    "Type": ["Numeric" if field in NUMERIC_FEATURES else "Categorical" for field in FEATURES],
    "Description": [definitions[field] for field in FEATURES],
})
st.subheader(f"Model inputs ({len(FEATURES)} fields, including engineered speed fields)")
st.dataframe(feature_table, hide_index=True, alt="Predictor columns used by the fitted SAFER model")
st.caption("The scenario form asks for 13 conditions. Speed excess and the speeding indicator are derived from the two speed entries. Auxiliary simulator_noise, simulator_risk_score, and severity columns are label-audit fields and are never predictors.")

st.subheader("Research contribution and boundaries")
st.markdown(
    "SAFER's contribution is an integrated, reproducible evaluation of entropy-adaptive ensemble weighting, validation-selected calibration and Fatal thresholding, local SHAP explanations, and a severe-imbalance stress sample. These components are established methods; the study does not claim a new base algorithm. The earlier beta comparison did not show a clear improvement over uniform soft voting, so adaptive weighting is treated as an evaluated research question rather than a proven advantage.\n\n"
    "The datasets are synthetic simulator draws. Dataset B is a separate draw from the same simulator. Dataset C is a 2,000-row no-replacement subset of B with 1,503 Slight, 470 Serious, and 27 Fatal records (1.35%, the nearest whole-row approximation to a 1.37% Fatal prior). B and C therefore overlap. Neither validates performance on real roads. Earlier STATS19-style experiments that used vehicle count do not establish a pre-crash result; a real-data study needs the same predictor-availability audit and new evaluation using only defensible pre-event fields."
)
st.markdown("**Interpretation:** the output is severity conditional on a hypothetical collision having occurred, not the probability that a collision will occur. The simulated speed values and model explanations are not causal evidence.")

results = load_experiment_table("ensemble_comparison.csv")
selected = results.loc[results["Model"].str.startswith("Selected SAFER")].set_index("Dataset")
refit = load_refit_metrics()
with st.container(horizontal=True):
    st.metric("Dataset A held-out accuracy", as_percent(selected.loc["Dataset A test", "accuracy"]), border=True)
    st.metric("Dataset A macro-F1", as_percent(selected.loc["Dataset A test", "macro_f1"]), border=True)
    st.metric("Dataset C Fatal recall", as_percent(selected.loc["Dataset C prior-shift", "fatal_recall"]), border=True)
st.caption(f"Model artifact: beta={refit['selection']['selected_beta']:g}, calibration={refit['selection']['selected_calibration']}, Fatal alert cutoff={refit['selection']['fatal_threshold_selected_on_validation']:.2f}. Metrics describe the synthetic label rule only.")
