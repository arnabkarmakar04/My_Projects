import streamlit as st
import pandas as pd

@st.cache_data
def load_data(uploaded_file):
    """Loads a CSV file into a pandas DataFrame without modifying data types."""
    try:
        return pd.read_csv(uploaded_file, low_memory=False)
    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded file is empty. Please upload a valid CSV file.")
        return None
    except pd.errors.ParserError:
        st.error("Error: The file could not be parsed. Please ensure it is a well-formatted CSV.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def init_session_state(uploaded_file):
    """Initializes the session state with the uploaded data."""
    if "original_data" not in st.session_state:
        st.session_state.original_data = load_data(uploaded_file)
        if st.session_state.original_data is not None:
            st.session_state.processed_data = st.session_state.original_data.copy()
            st.session_state.data_history = []
            st.session_state.redo_history = []
            st.session_state.selected_column_for_rename = None
            st.success("Session initialized.")

def backup_data():
    """Creates a backup of the current processed data state."""
    st.session_state.data_history.append(st.session_state.processed_data.copy())
