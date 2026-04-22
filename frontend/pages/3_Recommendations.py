import streamlit as st

from frontend.utils.page_helpers import load_data_or_stop


st.set_page_config(
    page_title="Recommendations",
    page_icon="R",
    layout="wide",
)

st.title("Recommendations")
st.caption("Recommendation engine placeholder.")

data = load_data_or_stop()

st.info("The recommendation engine will be connected next.")

kpi_columns = st.columns(3)
kpi_columns[0].metric("Products Available", f"{len(data['products']):,}")
kpi_columns[1].metric("Inventory Rows", f"{len(data['inventory']):,}")
kpi_columns[2].metric("Sales Rows", f"{len(data['sales']):,}")

st.subheader("Planned Inputs")
st.write(
    "This page will use current inventory, sales velocity, reorder thresholds, "
    "store demand, and supplier data once the recommendation logic is added."
)
