import streamlit as st
from utils.helpers import init_session_state
from components.overview import display_data_overview
from components.cleaning import display_cleaning
from components.visualization import display_visualizations
from components.analysis import display_advanced_analytics
from components.time_series import display_time_series_analysis

st.set_page_config(
    page_title="Data Spectra",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("✨ Data Spectra")
st.write("**By: Arnab** | An Interactive EDA Tool")
st.markdown("---")

with st.container(border=True):
    st.header("Get Started: Upload Your Dataset")
    st.markdown("""
        Welcome to DataSpectra! To begin your analysis, please upload a CSV file. 
        This tool will help you clean, visualize, and analyze your data to uncover hidden insights.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/256/8242/8242984.png", width=200)

    with col2:
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here or click to browse.", 
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_file is None:
            st.info("📂 Waiting for you to upload a CSV file...")
        else:
            st.success("✅ File uploaded successfully! Your analysis is ready below.")


if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.clear()
        init_session_state(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if st.session_state.get("original_data") is not None and not st.session_state.original_data.empty:
        display_data_overview(st.session_state.processed_data)
        display_cleaning()
        filtered_data = display_visualizations(st.session_state.processed_data)
        display_advanced_analytics(filtered_data)
        display_time_series_analysis(filtered_data)
    
    else:
        st.error("The uploaded file could not be processed. Please try another file. 😔")
