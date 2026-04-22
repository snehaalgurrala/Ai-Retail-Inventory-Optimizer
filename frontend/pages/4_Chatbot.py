import streamlit as st

from frontend.utils.page_helpers import load_data_or_stop


st.set_page_config(
    page_title="Chatbot",
    page_icon="C",
    layout="wide",
)

st.title("Chatbot")
st.caption("Business question assistant placeholder.")

load_data_or_stop()

st.info("The chatbot interface will be connected after the core analytics are ready.")

st.subheader("Sample Business Questions")
st.write("Try questions like these when the chatbot is connected:")

questions = [
    "Which products are close to reorder threshold?",
    "Which store has the highest sales quantity?",
    "What are the top selling products this month?",
    "Which suppliers support the most products?",
    "Which categories have the lowest current stock?",
]

for question in questions:
    st.markdown(f"- {question}")
