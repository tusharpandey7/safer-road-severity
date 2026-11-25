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
    "driver_age_band": "Driver age band",
    "actual_speed_kmh_simulated": "Hypothetical pre-impact speed",
    "speed_excess_kmh_simulated": "Derived speed excess",
    "speeding_indicator_simulated": "Derived speeding indicator",
}


def options_for(data: pd.DataFrame, column: str) -> list:
    return sorted(data[column].dropna().unique().tolist())


st.title("Scenario lab", icon=":material/analytics:")
st.caption("Enter a hypothetical pre-crash scenario. SAFER estimates severity conditional on a collision and explains its score using SHAP.")
st.warning(
    "All records, labels, and pre-impact speed values are synthetic. Speed is a hypothetical simulator input, not a measured vehicle speed. This estimates severity conditional on a collision; it does not estimate whether a collision will occur.",
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
        speed = st.selectbox("Posted speed limit (km/h)", speed_options, index=speed_options.index(int(typical["speed_limit_kmh"])), help="The posted legal limit, not the vehicle's speed.")
        actual_speed = st.slider("Hypothetical pre-impact vehicle speed (km/h)", 0, 160, int(typical["actual_speed_kmh_simulated"]), step=5, help="A simulated value for this synthetic scenario. It is not observed or estimated from real telematics.")
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
        "driver_age_band": age_band,
        "actual_speed_kmh_simulated": int(actual_speed),
        "speed_excess_kmh_simulated": max(0, int(actual_speed) - int(speed)),
        "speeding_indicator_simulated": int(int(actual_speed) > int(speed)),
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
            st.markdown("**Calibrated class scores**")
            score_frame = pd.DataFrame({"Severity": LABELS, "Score": [float(value) * 100 for value in result["scores"]]})
            probability_chart = (
                alt.Chart(score_frame)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Score:Q", title="Calibrated score (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Severity:N", sort=LABELS, title=None),
                    color=alt.Color("Severity:N", scale=alt.Scale(domain=LABELS, range=["#138A80", "#D69E2E", "#D9534F"]), legend=None),
                    tooltip=["Severity:N", alt.Tooltip("Score:Q", format=".1f")],
                ).properties(height=150)
            )
            st.altair_chart(probability_chart, alt="SAFER model-assigned scores for Slight, Serious, and Fatal")
    predictor = load_model()
    st.caption(f"Decision rule: Fatal is selected at or above the validation-selected {predictor.fatal_threshold:.0%} threshold; otherwise the higher-scoring non-Fatal class is returned. Calibration method: {predictor.calibration_method}. Scores and speed values are synthetic and are not real-world probability guarantees.")

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
        narrative = f"For this entered scenario, the adaptive ensemble selected **{predicted}** with a {selected_score:.1%} calibrated score for the synthetic study distribution."
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
            st.dataframe(shown, hide_index=True, alt="Scenario inputs and derived synthetic speed features with their local SHAP contributions")
    else:
        st.warning("The prediction is available, but the local explanation returned an unexpected output shape.")

    with st.expander("Compare a second hypothetical scenario"):
        st.caption("The alternative starts from the submitted scenario. Edit any conditions below; unedited fields are held constant. Differences are model comparisons, not causal estimates.")
        base = result["scenario"]
        alternative = dict(base)
        c1, c2, c3 = st.columns(3)
        with c1:
            alternative["day_type"] = st.selectbox("Alternative day type", options_for(data, "day_type"), index=options_for(data, "day_type").index(base["day_type"]), key="alt_day")
            alternative["hour"] = st.slider("Alternative hour", 0, 23, int(base["hour"]), key="alt_hour")
            alternative["speed_limit_kmh"] = st.selectbox("Alternative posted limit", speed_options, index=speed_options.index(int(base["speed_limit_kmh"])), key="alt_limit")
            alternative["actual_speed_kmh_simulated"] = st.slider("Alternative hypothetical speed", 0, 160, int(base["actual_speed_kmh_simulated"]), step=5, key="alt_speed")
            alternative["road_type"] = st.selectbox("Alternative road type", options_for(data, "road_type"), index=options_for(data, "road_type").index(base["road_type"]), key="alt_road")
        with c2:
            alternative["junction_detail"] = st.selectbox("Alternative junction", options_for(data, "junction_detail"), index=options_for(data, "junction_detail").index(base["junction_detail"]), key="alt_junction")
            alternative["junction_control"] = st.selectbox("Alternative junction control", options_for(data, "junction_control"), index=options_for(data, "junction_control").index(base["junction_control"]), key="alt_control")
            alternative["urban_or_rural"] = st.selectbox("Alternative area", options_for(data, "urban_or_rural"), index=options_for(data, "urban_or_rural").index(base["urban_or_rural"]), key="alt_area")
            alternative["weather_conditions"] = st.selectbox("Alternative weather", options_for(data, "weather_conditions"), index=options_for(data, "weather_conditions").index(base["weather_conditions"]), key="alt_weather")
        with c3:
            alternative["light_conditions"] = st.selectbox("Alternative lighting", options_for(data, "light_conditions"), index=options_for(data, "light_conditions").index(base["light_conditions"]), key="alt_light")
            alternative["road_surface_conditions"] = st.selectbox("Alternative surface", options_for(data, "road_surface_conditions"), index=options_for(data, "road_surface_conditions").index(base["road_surface_conditions"]), key="alt_surface")
            alternative["vehicle_type_involved"] = st.selectbox("Alternative vehicle type", options_for(data, "vehicle_type_involved"), index=options_for(data, "vehicle_type_involved").index(base["vehicle_type_involved"]), key="alt_vehicle")
            alternative["driver_age_band"] = st.selectbox("Alternative driver age band", options_for(data, "driver_age_band"), index=options_for(data, "driver_age_band").index(base["driver_age_band"]), key="alt_age")
        alternative["speed_excess_kmh_simulated"] = max(0, int(alternative["actual_speed_kmh_simulated"]) - int(alternative["speed_limit_kmh"]))
        alternative["speeding_indicator_simulated"] = int(int(alternative["actual_speed_kmh_simulated"]) > int(alternative["speed_limit_kmh"]))
        if st.button("Compare scenarios", key="compare_scenarios", type="primary"):
            model = load_model()
            alternative_row = pd.DataFrame([alternative], columns=FEATURES)
            alternative_scores = model.predict_proba(alternative_row)[0]
            alternative_prediction = str(model.predict(alternative_row)[0])
            comparison_frame = pd.DataFrame({
                "Severity": LABELS,
                "Submitted scenario": np.asarray(result["scores"]) * 100,
                "Alternative scenario": alternative_scores * 100,
            })
            comparison_frame["Change (percentage points)"] = comparison_frame["Alternative scenario"] - comparison_frame["Submitted scenario"]
            st.markdown(f"Submitted: **{result['predicted']}** · Alternative: **{alternative_prediction}**")
            st.dataframe(comparison_frame.style.format({"Submitted scenario": "{:.1f}%", "Alternative scenario": "{:.1f}%", "Change (percentage points)": "{:+.1f}"}), hide_index=True, alt="Class score differences for two synthetic scenarios")
            st.caption("These are model score changes under the simulator. They do not show that changing a factor would cause a real-world change in severity.")

    download = {
        "predicted_severity": predicted,
        "synthetic_study_calibrated_class_scores": dict(zip(LABELS, result["scores"])),
        "scenario": result["scenario"],
        "calibration_scope": "Calibrated on synthetic Dataset A validation data; not a real-world probability guarantee.",
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
