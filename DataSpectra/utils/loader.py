import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None