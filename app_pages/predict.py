"""Interactive scenario scoring and SHAP explanation."""
from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.model_service import FEATURES, LABELS, MODEL_PATH
from src.ui_helpers import load_data, load_model


FEATURE_LABELS = {
    "day_type": "Day type",
    "hour": "Hour of day",
    "speed_limit_kmh": "Posted speed limit",
    "road_type": "Road type",
    "junction_detail": "Junction detail",
    "junction_control": "Junction control",
    "urban_or_rural": "Urban or rural setting",
    "weather_conditions": "Weather",
    "light_conditions": "Light conditions",
    "road_surface_conditions": "Road surface",
    "vehicle_type_involved": "Vehicle type",
    "number_of_vehicles": "Vehicles involved",
    "driver_age_band": "Driver age band",
}


def options_for(data: pd.DataFrame, column: str) -> list:
    return sorted(data[column].dropna().unique().tolist())


st.title("Scenario lab", icon=":material/analytics:")
st.caption("Enter a collision scenario. SAFER returns a model-selected severity class and explains that score using SHAP.")
st.warning(
    "This form classifies severity conditional on a collision scenario. It does not estimate the chance that a crash will occur. The input data and labels are synthetic.",
    title="Prototype scope",
)

data = load_data("A")
typical = data.mode(dropna=True).iloc[0]

with st.form("scenario_form", border=True):
    st.subheader("Time and road context")
    time_col, road_col = st.columns(2, gap="large")
    with time_col:
        day_type = st.selectbox("Day type", options_for(data, "day_type"), index=options_for(data, "day_type").index(typical["day_type"]))
        hour = st.slider("Local hour", min_value=0, max_value=23, value=int(typical["hour"]), help="Uses a 24-hour clock. Collision date is excluded by the SAFER model specification.")
        speed_options = sorted(int(value) for value in data["speed_limit_kmh"].dropna().unique())
        speed = st.selectbox("Posted speed limit (km/h)", speed_options, index=speed_options.index(int(typical["speed_limit_kmh"])))
        road_type = st.selectbox("Road type", options_for(data, "road_type"), index=options_for(data, "road_type").index(typical["road_type"]))
    with road_col:
        junction_detail = st.selectbox("Junction detail", options_for(data, "junction_detail"), index=options_for(data, "junction_detail").index(typical["junction_detail"]))
        junction_control = st.selectbox("Junction control", options_for(data, "junction_control"), index=options_for(data, "junction_control").index(typical["junction_control"]))
        urban_rural = st.selectbox("Area setting", options_for(data, "urban_or_rural"), index=options_for(data, "urban_or_rural").index(typical["urban_or_rural"]))

    st.subheader("Weather, lighting, and surface")
    weather_col, surface_col = st.columns(2, gap="large")
    with weather_col:
        weather = st.selectbox("Weather conditions", options_for(data, "weather_conditions"), index=options_for(data, "weather_conditions").index(typical["weather_conditions"]))
        light = st.selectbox("Light conditions", options_for(data, "light_conditions"), index=options_for(data, "light_conditions").index(typical["light_conditions"]))
    with surface_col:
        surface = st.selectbox("Road surface", options_for(data, "road_surface_conditions"), index=options_for(data, "road_surface_conditions").index(typical["road_surface_conditions"]))
        vehicles = st.number_input(
            "Number of vehicles involved", min_value=1,
            max_value=max(1, int(data["number_of_vehicles"].max())),
            value=int(typical["number_of_vehicles"]), step=1,
        )

    st.subheader("Vehicle and driver context")
    vehicle_col, age_col = st.columns(2, gap="large")
    with vehicle_col:
        vehicle_type = st.selectbox("Vehicle type involved", options_for(data, "vehicle_type_involved"), index=options_for(data, "vehicle_type_involved").index(typical["vehicle_type_involved"]))
    with age_col:
        age_band = st.selectbox("Driver age band", options_for(data, "driver_age_band"), index=options_for(data, "driver_age_band").index(typical["driver_age_band"]))

    submitted = st.form_submit_button(
        "Generate severity assessment", type="primary",
        icon=":material/bolt:", width="stretch",
    )

if submitted:
    scenario = {
        "day_type": day_type,
        "hour": int(hour),
        "speed_limit_kmh": int(speed),
        "road_type": road_type,
        "junction_detail": junction_detail,
        "junction_control": junction_control,
        "urban_or_rural": urban_rural,
        "weather_conditions": weather,
        "light_conditions": light,
        "road_surface_conditions": surface,
        "vehicle_type_involved": vehicle_type,
        "number_of_vehicles": int(vehicles),
        "driver_age_band": age_band,
    }
    row = pd.DataFrame([scenario], columns=FEATURES)
    try:
        predictor = load_model()
        with st.spinner("Scoring the scenario and computing local SHAP values…"):
            scores = predictor.predict_proba(row)[0]
            predicted = str(predictor.predict(row)[0])
            shap_values, shap_base = predictor.explain(row)
        st.session_state["safer_latest_result"] = {
            "scenario": scenario,
            "scores": scores.tolist(),
            "predicted": predicted,
            "shap_values": np.asarray(shap_values).tolist(),
            "shap_base": np.asarray(shap_base).tolist(),
        }
    except FileNotFoundError:
        st.error("The fitted model file is missing. From the project folder, run `python src/train_model.py` and refresh this page.", title="Model artifact not found")
    except Exception as exc:
        st.error(f"Could not score this scenario: {exc}", title="Prediction could not be completed")

result = st.session_state.get("safer_latest_result")
if result:
    st.subheader("Prediction")
    st.caption("Showing the most recently submitted scenario.")
    predicted = result["predicted"]
    target_idx = LABELS.index(predicted)
    selected_score = float(result["scores"][target_idx])
    with st.container(border=True):
        result_col, rule_col = st.columns([1, 1.5], vertical_alignment="center")
        with result_col:
            color = {"Slight": "green", "Serious": "orange", "Fatal": "red"}[predicted]
            st.badge(f"{predicted} severity", color=color, icon=":material/traffic:")
            st.metric("Selected class score", f"{selected_score:.1%}", border=True)
        with rule_col:
            st.markdown("**Three-class model scores**")
            score_frame = pd.DataFrame({"Severity": LABELS, "Score": [float(value) * 100 for value in result["scores"]]})
            probability_chart = (
                alt.Chart(score_frame)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Score:Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Severity:N", sort=LABELS, title=None),
                    color=alt.Color("Severity:N", scale=alt.Scale(domain=LABELS, range=["#138A80", "#D69E2E", "#D9534F"]), legend=None),
                    tooltip=["Severity:N", alt.Tooltip("Score:Q", format=".1f")],
                ).properties(height=150)
            )
            st.altair_chart(probability_chart, alt="SAFER model-assigned scores for Slight, Serious, and Fatal")
    st.caption(f"Fatal decision rule: classify as Fatal when its model score is at least {load_model().fatal_threshold:.0%}; otherwise select the higher-scoring non-Fatal class. Scores are not calibrated real-world probabilities.")

    st.subheader("Plain-English explanation")
    attribution = np.asarray(result["shap_values"], dtype=float)
    if attribution.ndim == 1:
        attribution = attribution[:, None]
    if attribution.shape[0] == len(FEATURES) and attribution.shape[1] == len(LABELS):
        class_effect = attribution[:, target_idx] * 100
        explanation = pd.DataFrame({
            "Feature": [FEATURE_LABELS[name] for name in FEATURES],
            "Entered value": [str(result["scenario"][name]) for name in FEATURES],
            "SHAP contribution (percentage points)": class_effect,
        })
        explanation["Direction"] = np.where(class_effect >= 0, "Raises this class score", "Lowers this class score")
        positive = explanation[explanation["SHAP contribution (percentage points)"] > 0].nlargest(3, "SHAP contribution (percentage points)")
        negative = explanation[explanation["SHAP contribution (percentage points)"] < 0].nsmallest(3, "SHAP contribution (percentage points)")
        def factor_list(frame: pd.DataFrame) -> str:
            return ", ".join(
                f"{row['Feature']} ({row['Entered value']}, {row['SHAP contribution (percentage points)']:+.1f} pp)"
                for _, row in frame.iterrows()
            )
        narrative = f"For this entered scenario, the adaptive ensemble selected **{predicted}** with a {selected_score:.1%} model score."
        if not positive.empty:
            narrative += f" The largest local signals pushing the {predicted} score upward were {factor_list(positive)}."
        if not negative.empty:
            narrative += f" Signals pulling that score downward were {factor_list(negative)}."
        st.markdown(narrative)
        st.caption("SHAP (SHapley Additive exPlanations) estimates how each entered field contributes to this model's selected-class score relative to a background sample. The attribution explains the model; it does not show that a factor causes an accident outcome.")

        bar = (
            alt.Chart(explanation)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("SHAP contribution (percentage points):Q", title=f"Contribution to {predicted} score (percentage points)"),
                y=alt.Y("Feature:N", sort=alt.SortField("SHAP contribution (percentage points)", order="ascending"), title=None),
                color=alt.condition(
                    alt.datum["SHAP contribution (percentage points)"] >= 0,
                    alt.value("#138A80"), alt.value("#D97706"),
                ),
                tooltip=["Feature:N", "Entered value:N", alt.Tooltip("SHAP contribution (percentage points):Q", format="+.2f"), "Direction:N"],
            ).properties(height=370)
        )
        st.altair_chart(bar, alt=f"SHAP contributions for each input to the {predicted} class score")
        with st.expander("See all input factors and SHAP contributions"):
            shown = explanation.copy()
            shown["SHAP contribution (percentage points)"] = shown["SHAP contribution (percentage points)"].map(lambda v: f"{v:+.2f}")
            st.dataframe(shown, hide_index=True, alt="All thirteen scenario inputs and their local SHAP contributions")
    else:
        st.warning("The prediction is available, but the local explanation returned an unexpected output shape.")

    download = {
        "predicted_severity": predicted,
        "model_scores_not_calibrated_probabilities": dict(zip(LABELS, result["scores"])),
        "scenario": result["scenario"],
        "explanation": "Local permutation SHAP contributions are model associations, not causes.",
    }
    st.download_button(
        "Download this assessment as JSON",
        data=json.dumps(download, indent=2),
        file_name="safer_scenario_assessment.json",
        mime="application/json",
        icon=":material/download:",
        width="content",
    )
