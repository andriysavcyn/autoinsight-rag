import streamlit as st
from controllers.rag_pipeline import setup_rag_pipeline
from views.chat_interface import render_ui  

st.set_page_config(page_title="AutoInsight AI", page_icon="🚘", layout="centered")

@st.cache_resource(show_spinner=False)
def load_ai_backend():
    try:
        return setup_rag_pipeline()
    except Exception as e:
        st.error(f"AI initialization error: {e}")
        return None

with st.spinner("Running neural networks..."):
    chain = load_ai_backend()

if chain is not None:
    render_ui(chain)
else:
    st.stop()