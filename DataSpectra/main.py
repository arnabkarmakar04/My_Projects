import streamlit as st
from utils.loader import load_data
from utils.overview import show_overview
from utils.recommend import recommend_visualization
from utils.visualizer import plot_gallery
from utils.analysis import run_analysis

st.set_page_config(page_icon=":bar_chart:", layout="wide")
st.title("Data Spectra")
st.write("**By: Arnab**")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None and not data.empty:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        show_overview(data, numeric_columns, categorical_columns)

        st.subheader("Visualization Recommendation")
        recommended_visualizations = recommend_visualization(data)
        st.write("Possible visualizations from your Data")
        st.write(recommended_visualizations)
        selected_plots = st.multiselect("**Select Visualization**", recommended_visualizations)

        plot_gallery(data, numeric_columns, categorical_columns, selected_plots)

        st.subheader("Advanced Analytical Features")
        run_analysis(data, numeric_columns)