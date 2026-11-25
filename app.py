"""SAFER interactive road-collision severity research prototype."""
import streamlit as st

st.set_page_config(
    page_title="SAFER | Severity lab",
    page_icon=":material/alt_route:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "SAFER is a synthetic-data college research prototype."},
)

pages = [
    st.Page(
        "app_pages/home.py", title="Overview", icon=":material/dashboard:", default=True,
        help="See what SAFER does, the reported benchmark scores, and the synthetic dataset scope.",
    ),
    st.Page(
        "app_pages/predict.py", title="Scenario lab", icon=":material/analytics:",
        help="Enter the 13 recorded scenario fields to get a three-class severity score and local SHAP explanation.",
    ),
    st.Page(
        "app_pages/performance.py", title="Model performance", icon=":material/monitoring:",
        help="Compare the supplied Dataset A and Dataset B metrics, class reports, confusion matrices, and ablation results.",
    ),
    st.Page(
        "app_pages/explorer.py", title="Dataset explorer", icon=":material/table_chart:",
        help="Explore and download the two supplied synthetic samples with simple severity breakdowns.",
    ),
    st.Page(
        "app_pages/methodology.py", title="Method and limits", icon=":material/menu_book:",
        help="Review the simulator, feature list, model rule, source documents, and research limitations.",
    ),
]

selected_page = st.navigation(pages, position="top")

with st.sidebar:
    st.markdown("## :material/alt_route: SAFER")
    st.caption("Road severity research prototype")
    st.badge("Synthetic study", icon=":material/science:", color="orange")
    st.markdown("**Model**")
    st.caption("Entropy-adaptive soft-voting ensemble")
    st.markdown("**Data**")
    st.caption("10,000 development rows · 5,000 independent-sample rows")
    st.caption("Severity is classified for an entered collision scenario. The app does not predict whether a crash will occur.")

selected_page.run()
