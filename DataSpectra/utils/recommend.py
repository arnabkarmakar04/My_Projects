def recommend_visualization(df):
    num_numeric_cols = len(df.select_dtypes(include=['float64', 'int64']).columns)
    num_categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
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
        recommended_vis.append("Line Chart")
    return recommended_vis
