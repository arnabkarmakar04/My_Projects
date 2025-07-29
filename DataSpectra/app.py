import streamlit as st
from utils.helpers import init_session_state
from components.overview import display_data_overview
from components.cleaning import display_cleaning
from components.visualization import display_visualizations
from components.analysis import display_advanced_analytics

st.set_page_config(
    page_title="Data Spectra",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Data Spectra")
st.write("**By: Arnab** | An Interactive EDA Tool")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your CSV file to begin analysis", 
    type=["csv"]
)

if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.clear()
        init_session_state(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if st.session_state.get("original_data") is not None and not st.session_state.original_data.empty:
        display_data_overview(st.session_state.processed_data)
        display_cleaning()
        display_visualizations(st.session_state.processed_data)
        display_advanced_analytics(st.session_state.processed_data)
    
    else:
        st.error("The uploaded file could not be processed. Please try another file.")
else:
    st.info("Awaiting for a CSV file to be uploaded.")
