import streamlit as st
import pandas as pd
import numpy as np

def display_data_overview(data):
    """Displays the data preview, shape, dtypes, and descriptive statistics."""
    st.subheader("About Dataset")
    
    with st.expander("Data Preview"):
        rows = st.slider(
            "Number of rows to display", 
            min_value=1, 
            max_value=len(data), 
            value=5, 
            key="slider_rows"
        )
        st.dataframe(data.head(rows))

    with st.expander("Dataset Information"):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("##### Column Data Types")
            st.dataframe(pd.DataFrame({
                "Column": data.columns, 
                "Data Type": [str(dtype) for dtype in data.dtypes]
            }))
            st.write(f"**Total Rows:** {data.shape[0]} | **Total Columns:** {data.shape[1]}")

        with col2:
            st.write("**Numerical Columns**")
            st.write(numeric_columns if numeric_columns else "None")
            
            if boolean_columns:
                st.write("**Boolean Columns**")
                st.write(boolean_columns)

        with col3:
            st.write("**Categorical Columns**")
            st.write(categorical_columns if categorical_columns else "None")

            if datetime_columns:
                st.write("**Datetime Columns**")
                st.write(datetime_columns)

    with st.expander("Show Descriptive Statistics"):
        st.write(data.describe())
        if data.isna().any().any():
            st.info("🔔 Missing values were detected in the dataset.")
