import streamlit as st
import pandas as pd

def show_overview(data, numeric_columns, categorical_columns):
    st.subheader("About Dataset")

    with st.expander("Data Preview"):
        rows = st.slider("Number of rows", min_value=1, max_value=len(data), value=5)
        st.dataframe(data.head(rows))

    with st.expander("Dataset Information"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write("Column Data Types")
            st.dataframe(pd.DataFrame({"Column": data.columns, "Data Type": [str(dtype) for dtype in data.dtypes]}))
            st.write(f"**Total Rows:** {data.shape[0]} | **Total Columns:** {data.shape[1]}")
        with col2:
            st.write("Numerical Columns")
            st.write(numeric_columns if numeric_columns else "None")
        with col3:
            st.write("Categorical Columns")
            st.write(categorical_columns if categorical_columns else "None")

    with st.expander("Show Descriptive Statistics"):
        st.write(data.describe())
        if data.isna().any().any():
            st.write("🔔 Missing values detected & filled with the median for numeric columns.")
            data[numeric_columns] = data[numeric_columns].apply(lambda col: col.fillna(col.median()), axis=0)