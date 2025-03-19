import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_icon=":bar_chart:", layout="wide")
st.title("Data Spectra")
st.write("**By: Arnab**")

@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, low_memory= False)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None and not data.empty:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        st.subheader("About Dataset")

        with st.expander("Data Preview"):
            rows = st.slider("Number of rows", min_value=1, max_value=len(data), value=5)
            st.dataframe(data.head(rows))

        with st.expander("Dataset Information"):
            col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write("Column Data Types")
            st.dataframe(pd.DataFrame({"Column": data.columns, "Data Type": data.dtypes.values}))
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

        st.subheader("Visualization Recommendation")    
        def recommend_visualization(df):
            num_numeric_cols = len(df.select_dtypes(include=['float64', 'int64']).columns)
            num_categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            num_rows = len(df)
            recommended_vis = []       
            if num_numeric_cols > 1:
                recommended_vis.extend(["Scatter Plot", "Heatmap"])
            if num_categorical_cols > 0 and num_numeric_cols > 0:
                recommended_vis.append("Bar Chart")
            if num_numeric_cols > 0:
                recommended_vis.append("Histogram")
            if num_categorical_cols > 0:
                recommended_vis.append("Donut Chart")
            if any(df.dtypes[col] == 'datetime64[ns]' for col in df.columns):
                recommended_vis.add("Line Chart")           
            return recommended_vis

        recommended_visualizations = recommend_visualization(data)
        st.write("Possible visualizations from your Data")
        st.write(recommended_visualizations)    
        selected_plots = st.multiselect("**Select Visualization**", recommended_visualizations)

        def plot_gallery():
            for plot_type in selected_plots:
                if plot_type == "Bar Chart":
                    st.write("Bar Chart")
                    bar_col = st.selectbox("Select a categorical column", categorical_columns, key="bar_chart")
                    bar_color = st.color_picker("Choose a color", "#FFA500", key="bar_color")
                    if bar_col:
                        fig = px.bar(data, x=bar_col, color_discrete_sequence=[bar_color], title=f'Bar Chart of {bar_col}')
                        st.plotly_chart(fig)

                elif plot_type == "Histogram":
                    st.write("Histogram")
                    hist_col = st.selectbox("Select a numeric column", numeric_columns, key="histogram")
                    hist_color = st.color_picker("Choose a color", "#FFA500", key="hist_color")
                    if hist_col:
                        fig = px.histogram(data, x=hist_col, color_discrete_sequence=[hist_color], title=f'Histogram of {hist_col}', nbins= 20)
                        fig.update_layout(bargap=0.2)
                        st.plotly_chart(fig)

                elif plot_type == "Scatter Plot":
                    st.write("Scatter Plot")
                    x_column = st.selectbox("Select X-axis column", numeric_columns, key="scatter_x")
                    y_column = st.selectbox("Select Y-axis column", numeric_columns, key="scatter_y")
                    if x_column and y_column:
                        fig = px.scatter(data, x=x_column, y=y_column, title="Scatter Plot")
                        st.plotly_chart(fig)

                elif plot_type == "Donut Chart":
                    st.write("Donut Chart")
                    donut_column = st.selectbox("Select a categorical column", categorical_columns, key="donut_chart")
                    if donut_column:
                        values = data[donut_column].value_counts().reset_index()
                        values.columns = [donut_column, "count"]
                        fig = px.pie(values, names=donut_column, values="count", hole=0.4, title="Donut Chart")
                        st.plotly_chart(fig)

                elif plot_type == "Heatmap":
                    st.write("Heatmap")
                    numeric_data = data.select_dtypes(include=["number"])
                    if numeric_data.empty:
                        st.warning("No numeric columns available for correlation.")
                    else:
                        corr_matrix = numeric_data.corr()
                        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='viridis', title="Heatmap", width=800, height=600)
                        st.plotly_chart(fig)

                elif plot_type == "Line Chart":
                    st.write("Line Chart")
                    line_x_col = st.selectbox("Select X-axis column (should be datetime or categorical)", data.columns, key="line_x")
                    line_y_col = st.selectbox("Select Y-axis column (should be numeric)", numeric_columns, key="line_y")
                    if line_x_col and line_y_col:
                        fig = px.line(data, x=line_x_col, y=line_y_col, title=f'Line Chart: {line_y_col} vs {line_x_col}')
                        st.plotly_chart(fig)


        plot_gallery()

        st.subheader("Advanced Analytical Features")

        with st.expander("Select Analysis Methods"):
            analysis_methods = st.multiselect(
                "Choose analysis methods to perform:",
                ["Linear Regression", "Random Forest Classification", "K-Means Clustering"]
            )

        if "Linear Regression" in analysis_methods:
            st.write("Linear Regression")
            x_col = st.selectbox("Select X column for Linear Regression", ["None"] + numeric_columns)
            y_col = st.selectbox("Select Y column for Linear Regression", ["None"] + numeric_columns)
            if x_col != "None" and y_col != "None":
                try:
                    model = LinearRegression()
                    X = data[[x_col]].apply(lambda col: col.fillna(col.median()), axis=0).values.reshape(-1, 1)
                    y = data[y_col].fillna(data[y_col].median()).values
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    intercept = model.intercept_
                    coef = model.coef_[0]
                    equation = f"y = {coef:.2f}x + {intercept:.2f}"
                    st.write(f"Regression Equation: {equation}")
                    reg_fig = px.scatter(data, x=x_col, y=y_col, trendline="ols", title=f'Linear Regression: {y_col} vs {x_col}')
                    st.plotly_chart(reg_fig)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        if "Random Forest Classification" in analysis_methods:
            st.write("Classification using Random Forest")
            target_var = st.selectbox("Select Target Variable", ["None"] + numeric_columns)
            if target_var != "None":
                try:
                    X = data[numeric_columns].drop(columns=[target_var], errors='ignore')
                    X = X.apply(lambda col: col.fillna(col.median()), axis=0)  # CHANGED: Fill missing with median
                    y = data[target_var].fillna(data[target_var].median())
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
                    importance_df = importance_df.sort_values(by="Importance", ascending=False)
                    importance_fig = px.bar(
                        importance_df,
                        x="Feature",
                        y="Importance",
                        title="Feature Importance",
                        labels={'x': 'Features', 'y': 'Importance'}
                    )
                    st.plotly_chart(importance_fig)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        if "K-Means Clustering" in analysis_methods:
            st.write("K-Means Clustering")
            kmeans_cols = st.multiselect("Select atleast two Columns for Clustering", numeric_columns, key="kmeans_cols")
            if len(kmeans_cols) >= 2:
                try:
                    num_clusters = st.number_input("Enter Number of Clusters (k)", min_value=2, max_value=20, value=3, step=1, key="num_clusters_input")
                    clustering_data = data[kmeans_cols]
                    clustering_data = clustering_data.apply(lambda col: col.fillna(col.median()), axis=0)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(clustering_data)
                    cluster_data = data.copy()
                    cluster_data["Cluster"] = labels
                    cluster_fig = px.scatter(cluster_data, x=kmeans_cols[0], y=kmeans_cols[1], color=cluster_data["Cluster"].astype(str), title=f"K-Means Clustering (k={num_clusters})", labels={"Cluster": "Cluster Group"})
                    st.plotly_chart(cluster_fig)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
