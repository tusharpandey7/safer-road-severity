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
        help="Enter a pre-crash scenario, including synthetic hypothetical speed, and review class scores and local SHAP explanations.",
    ),
    st.Page(
        "app_pages/performance.py", title="Model performance", icon=":material/monitoring:",
        help="Compare all model configurations, paired confidence intervals, and calibration quality across test samples.",
    ),
    st.Page(
        "app_pages/explorer.py", title="Dataset explorer", icon=":material/table_chart:",
        help="Explore and download Datasets A, B, and the low-Fatal-prior Dataset C stress sample.",
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
    st.caption("10,000 Dataset A rows · 5,000 Dataset B rows · Dataset C stress sample")
    st.caption("Pre-crash severity conditional on a hypothetical collision; no collision-occurrence estimate.")

selected_page.run()
