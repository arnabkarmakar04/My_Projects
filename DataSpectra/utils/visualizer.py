import streamlit as st
import plotly.express as px

def plot_gallery(data, numeric_columns, categorical_columns, selected_plots):
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
                fig = px.histogram(data, x=hist_col, color_discrete_sequence=[hist_color], title=f'Histogram of {hist_col}', nbins=20)
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
