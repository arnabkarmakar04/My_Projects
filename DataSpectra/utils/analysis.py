import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd


def run_analysis(data, numeric_columns):
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
                X = X.apply(lambda col: col.fillna(col.median()), axis=0)
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
        kmeans_cols = st.multiselect("Select at least two Columns for Clustering", numeric_columns, key="kmeans_cols")
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
