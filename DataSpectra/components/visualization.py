import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_global_filter_widgets(data):
    st.sidebar.title("Global Data Filters")
    filter_container = st.sidebar.container()

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    all_filterable_columns = numeric_columns + categorical_columns + datetime_columns

    if not all_filterable_columns:
        return data

    filter_cols = filter_container.multiselect(
        "Select global columns to filter by:",
        options=all_filterable_columns,
        key="global_filter_cols"
    )

    filtered_data = data.copy()

    for col in filter_cols:
        if col in numeric_columns:
            min_val, max_val = float(filtered_data[col].min()), float(filtered_data[col].max())
            if min_val < max_val:
                slider_range = filter_container.slider(
                    f"Select range for '{col}'",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"global_slider_{col}"
                )
                filtered_data = filtered_data[
                    (filtered_data[col] >= slider_range[0]) & (filtered_data[col] <= slider_range[1])
                ]
            else:
                filter_container.info(f"Column '{col}' has only one unique value ({min_val}) and cannot be filtered with a range slider.")

        elif col in categorical_columns:
            unique_vals = sorted(filtered_data[col].astype(str).unique())
            selected_vals = filter_container.multiselect(
                f"Select values for '{col}'",
                options=unique_vals,
                key=f"global_multi_{col}"
            )
            if selected_vals:
                filtered_data = filtered_data[filtered_data[col].isin(selected_vals)]
        
        elif col in datetime_columns:
            min_date = filtered_data[col].min()
            max_date = filtered_data[col].max()
            if min_date.date() < max_date.date():
                date_range = filter_container.date_input(
                    f"Select date range for '{col}'",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key=f"global_date_{col}"
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_data = filtered_data[
                        (filtered_data[col].dt.date >= start_date) & (filtered_data[col].dt.date <= end_date)
                    ]
            else:
                filter_container.info(f"Column '{col}' contains dates from a single day and cannot be filtered with a date range.")

    return filtered_data

def render_local_filter_widgets(data, chart_key):
    st.sidebar.markdown("---")
    
    with st.sidebar.expander(f"Chart-Specific Filters for: {chart_key}", expanded=False):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        all_filterable_columns = numeric_columns + categorical_columns + datetime_columns

        if not all_filterable_columns:
            return data

        filter_cols = st.multiselect(
            "Select chart-specific columns to filter by:",
            options=all_filterable_columns,
            key=f"local_filter_cols_{chart_key}"
        )

        filtered_data = data.copy()

        for col in filter_cols:
            if col in numeric_columns:
                min_val, max_val = float(filtered_data[col].min()), float(filtered_data[col].max())
                if min_val < max_val:
                    slider_range = st.slider(
                        f"Select range for '{col}'",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"local_slider_{col}_{chart_key}"
                    )
                    filtered_data = filtered_data[
                        (filtered_data[col] >= slider_range[0]) & (filtered_data[col] <= slider_range[1])
                    ]
                else:
                    st.info(f"Column '{col}' has only one unique value ({min_val}) and cannot be filtered with a range slider.")

            elif col in categorical_columns:
                unique_vals = sorted(filtered_data[col].astype(str).unique())
                selected_vals = st.multiselect(
                    f"Select values for '{col}'",
                    options=unique_vals,
                    key=f"local_multi_{col}_{chart_key}"
                )
                if selected_vals:
                    filtered_data = filtered_data[filtered_data[col].isin(selected_vals)]
            
            elif col in datetime_columns:
                min_date = filtered_data[col].min()
                max_date = filtered_data[col].max()
                if min_date.date() < max_date.date():
                    date_range = st.date_input(
                        f"Select date range for '{col}'",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        key=f"local_date_{col}_{chart_key}"
                    )
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_data = filtered_data[
                            (filtered_data[col].dt.date >= start_date) & (filtered_data[col].dt.date <= end_date)
                        ]
                else:
                    st.info(f"Column '{col}' contains dates from a single day and cannot be filtered with a date range.")
        
        return filtered_data

def _render_bar_chart(data, plot_type, categorical_columns):
    bar_col = st.selectbox("Select a categorical column", ["Select a column"] + categorical_columns, key=f"bar_{plot_type}")
    if bar_col != "Select a column":
        bar_data = data[bar_col].value_counts().reset_index()
        bar_data.columns = [bar_col, 'Count']
        fig = px.bar(bar_data, x=bar_col, y='Count', title=f'Distribution of {bar_col}',
                     color=bar_col, text_auto=True,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

def _render_histogram(data, plot_type, numeric_columns):
    hist_col = st.selectbox("Select a numeric column", ["Select a column"] + numeric_columns, key=f"hist_{plot_type}")
    if hist_col != "Select a column":
        fig = px.histogram(data, x=hist_col, title=f'Histogram of {hist_col}',
                           marginal="rug",
                           color_discrete_sequence=['#636EFA'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

def _render_box_plot(data, plot_type, numeric_columns, categorical_columns):
    st.write("View the distribution of a numeric column, optionally grouped by a category.")
    col1, col2 = st.columns(2)
    with col1:
        y_col_box = st.selectbox("Select a numeric column (Y-axis)", ["Select a column"] + numeric_columns, key=f"boxplot_y_{plot_type}")
    with col2:
        x_col_box = st.selectbox("Group by a categorical column (X-axis, optional)", ["Select a column"] + categorical_columns, key=f"boxplot_x_{plot_type}")

    if y_col_box != "Select a column":
        x_val = None if x_col_box == "Select a column" else x_col_box
        color_arg_box = x_val
        fig_box = px.box(data, x=x_val, y=y_col_box, color=color_arg_box,
                         title=f"Box Plot of {y_col_box}" + (f" by {x_col_box}" if x_val is not None else ""),
                         points="outliers")
        if x_val is not None:
            fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

def _render_scatter_plot(data, plot_type, numeric_columns, categorical_columns):
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis", ["Select X-axis"] + numeric_columns, key=f"scatter_x_{plot_type}")
    
    y_col_options = ["Select Y-axis"] + [col for col in numeric_columns if col not in ["Select X-axis", x_col]]

    with col2:
        y_col = st.selectbox("Select Y-axis", y_col_options, key=f"scatter_y_{plot_type}")
    
    color_col = st.selectbox("Color by (optional)", ["Select a column"] + categorical_columns, key=f"scatter_color_{plot_type}")
    
    if x_col != "Select X-axis" and y_col != "Select Y-axis":
        color_arg = None if color_col == "Select a column" else color_col
        fig = px.scatter(data, x=x_col, y=y_col, color=color_arg,
                         title=f"Scatter Plot: {y_col} vs {x_col}",
                         hover_data=data.columns)
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

def _render_donut_chart(data, plot_type, categorical_columns):
    donut_col = st.selectbox("Select a categorical column", ["Select a column"] + categorical_columns, key=f"donut_{plot_type}")
    if donut_col != "Select a column":
        values = data[donut_col].value_counts().reset_index()
        values.columns = [donut_col, "count"]
        fig = px.pie(values, names=donut_col, values="count", hole=0.5,
                     title=f"Donut Chart of {donut_col}",
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(values))
        st.plotly_chart(fig, use_container_width=True)

def _render_heatmap(data, plot_type, numeric_columns):
    st.info("This heatmap shows the correlation between all numeric columns in the filtered data.")
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                        color_continuous_scale='Plasma',
                        title="Correlation Heatmap")
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Heatmap requires at least two numeric columns.")

def _render_line_chart(data, plot_type, numeric_columns):
    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    if not datetime_cols:
        st.warning("Line charts require a datetime column. Please convert a column to datetime format first.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        line_x = st.selectbox("Select X-axis (time)", ["Select X-axis"] + datetime_cols, key=f"line_x_{plot_type}")
    with col2:
        line_y = st.selectbox("Select Y-axis (value)", ["Select Y-axis"] + numeric_columns, key=f"line_y_{plot_type}")
    if line_x != "Select X-axis" and line_y != "Select Y-axis":
        line_data = data[[line_x, line_y]].dropna().sort_values(by=line_x)
        fig = px.line(line_data, x=line_x, y=line_y,
                      title=f'Line Chart: {line_y} over time',
                      markers=True)
        fig.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig, use_container_width=True)

def _render_aggregated_bar_chart(data, selected_plot, numeric_cols, categorical_cols):
    st.write("#### Aggregated Bar Chart")
    st.write("Compare an aggregated numeric value (like mean or sum) across different categories.")
    
    if not numeric_cols or not categorical_cols:
        st.warning("This plot requires at least one numeric and one categorical column.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_col_agg = st.selectbox("Select a categorical column (X-axis)", ["Select a column"] + categorical_cols, key=f"adv_cat_agg_{selected_plot}")
    with col2:
        num_col_agg = st.selectbox("Select a numeric column (Y-axis)", ["Select a column"] + numeric_cols, key=f"adv_num_agg_{selected_plot}")
    with col3:
        agg_func = st.selectbox("Select an aggregation function", ["Select a function", "mean", "sum", "median", "min", "max"], key=f"adv_agg_func_{selected_plot}")

    if cat_col_agg != "Select a column" and num_col_agg != "Select a column" and agg_func != "Select a function":
        try:
            agg_df = data.groupby(cat_col_agg)[num_col_agg].agg(agg_func).reset_index()
            fig_agg_bar = px.bar(agg_df, x=cat_col_agg, y=num_col_agg,
                                 title=f"{agg_func.capitalize()} of {num_col_agg} by {cat_col_agg}",
                                 text_auto='.2s')
            fig_agg_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_agg_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_grouped_line_chart(data, selected_plot, numeric_cols, categorical_cols):
    st.write("#### Grouped Line Chart")
    st.write("Compare trends over time for different categories.")

    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    if not datetime_cols or not numeric_cols or not categorical_cols:
        st.warning("This plot requires at least one datetime, one numeric, and one categorical column.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        line_x = st.selectbox("Select X-axis (time)", ["Select a column"] + datetime_cols, key=f"adv_line_x_{selected_plot}")
    with col2:
        line_y = st.selectbox("Select Y-axis (value)", ["Select a column"] + numeric_cols, key=f"adv_line_y_{selected_plot}")
    with col3:
        group_by = st.selectbox("Group by (color)", ["Select a column"] + categorical_cols, key=f"adv_line_group_{selected_plot}")

    if line_x != "Select a column" and line_y != "Select a column" and group_by != "Select a column":
        try:
            line_data = data[[line_x, line_y, group_by]].dropna().sort_values(by=line_x)
            fig_grouped_line = px.line(line_data, x=line_x, y=line_y, color=group_by,
                                       title=f'Line Chart: {line_y} over time by {group_by}',
                                       markers=True)
            fig_grouped_line.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_grouped_line, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_violin_plot(data, selected_plot, numeric_cols, categorical_cols):
    st.write("#### Categorical vs. Numeric Comparison")
    st.write("Use a Violin Plot to see the distribution of a numeric value across different categories.")
    
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Select a Categorical Column", ["Select a column"] + categorical_cols, key=f"adv_cat_col_{selected_plot}")
    with col2:
        num_col_for_violin = st.selectbox("Select a Numeric Column", ["Select a column"] + numeric_cols, key=f"adv_num_col_violin_{selected_plot}")

    if cat_col != "Select a column" and num_col_for_violin != "Select a column":
        try:
            fig_violin = px.violin(data, x=cat_col, y=num_col_for_violin, 
                                   color=cat_col,
                                   title=f"Distribution of '{num_col_for_violin}' by '{cat_col}'",
                                   box=True,
                                   points="all")
            fig_violin.update_layout(showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_enhanced_scatter_plot(data, selected_plot, numeric_cols, categorical_cols):
    st.write("#### Enhanced Scatter Plot")
    st.write("Use a Scatter Plot to compare up to four different columns at once.")

    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    with sc_col1:
        x_axis = st.selectbox("Select X-Axis (Numeric)", ["Select a column"] + numeric_cols, key=f"adv_x_axis_{selected_plot}")
    
    y_axis_options = ["Select a column"] + [col for col in numeric_cols if col != x_axis]
    with sc_col2:
        y_axis = st.selectbox("Select Y-Axis (Numeric)", y_axis_options, key=f"adv_y_axis_{selected_plot}")
    
    with sc_col3:
        color_axis = st.selectbox("Color By (Categorical)", ["Select a column"] + categorical_cols, key=f"adv_color_axis_{selected_plot}")

    used_cols = {x_axis, y_axis}
    size_axis_options = ["Select a column"] + [col for col in numeric_cols if col not in used_cols]
    with sc_col4:
        size_axis = st.selectbox("Size By (Numeric)", size_axis_options, key=f"adv_size_axis_{selected_plot}")

    if x_axis != "Select a column" and y_axis != "Select a column":
        try:
            color_arg = None if color_axis == "Select a column" else color_axis
            size_arg = None if size_axis == "Select a column" else size_axis
            fig_scatter = px.scatter(data, x=x_axis, y=y_axis, color=color_arg, size=size_arg,
                                     title=f"Scatter Plot: '{y_axis}' vs. '{x_axis}'",
                                     hover_data=data.columns)
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_one_vs_many_comparison(data, selected_plot, numeric_cols, categorical_cols):
    st.write("#### One-vs-Many Column Comparison")
    st.write("Use this to generate a series of scatter plots, comparing one main column against several others.")

    sm_col1, sm_col2, sm_col3 = st.columns(3)
    with sm_col1:
        main_col = st.selectbox("Select Main Column to Compare", ["Select a column"] + numeric_cols, key=f"adv_main_col_{selected_plot}")
    with sm_col2:
        other_cols = st.multiselect("Select Other Columns to Compare Against", [col for col in numeric_cols if col != main_col], key=f"adv_other_cols_{selected_plot}")
    color_sm_options = ["Select a column"] + [c for c in categorical_cols if c != main_col]
    with sm_col3:
        color_col_sm = st.selectbox("Color By (Categorical)", color_sm_options, key=f"adv_color_col_sm_{selected_plot}")

    if main_col != "Select a column" and len(other_cols) > 0:
        try:
            color_arg_sm = None if color_col_sm == "Select a column" else color_col_sm
            for i, other_col in enumerate(other_cols):
                if i % 2 == 0:
                    cols = st.columns(2)
                with cols[i % 2]:
                    fig = px.scatter(data, x=main_col, y=other_col, color=color_arg_sm,
                                     title=f"'{main_col}' vs. '{other_col}'")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_categorical_heatmap(data, selected_plot, categorical_cols):
    st.write("#### Categorical vs. Categorical Analysis")
    st.write("Use a Heatmap to see the relationship between two categorical variables.")

    if len(categorical_cols) < 2:
        st.warning("This plot requires at least two categorical columns in the data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cat1 = st.selectbox("Select the first categorical column", ["Select a column"] + categorical_cols, key=f"adv_cat1_{selected_plot}")
    
    cat2_options = ["Select a column"] + [c for c in categorical_cols if c != cat1]
    with col2:
        cat2 = st.selectbox("Select the second categorical column", cat2_options, key=f"adv_cat2_{selected_plot}")

    if cat1 != "Select a column" and cat2 != "Select a column":
        try:
            crosstab = pd.crosstab(data[cat1], data[cat2])
            fig_heatmap = px.imshow(crosstab, text_auto=True, aspect="auto",
                                    color_continuous_scale='Blues',
                                    title=f"Heatmap of {cat1} vs. {cat2}")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def _render_grouped_bar_chart(data, selected_plot, categorical_cols):
    st.write("#### Categorical vs. Categorical Analysis (Grouped Bar)")
    st.write("Use a **Grouped Bar Chart** to compare the frequency of categories in one column across another.")

    if len(categorical_cols) < 2:
        st.warning("This plot requires at least two categorical columns in the data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cat1_bar = st.selectbox("Select the main categorical column (X-axis)", ["Select a column"] + categorical_cols, key=f"adv_cat1_bar_{selected_plot}")
    
    cat2_bar_options = ["Select a column"] + [c for c in categorical_cols if c != cat1_bar]
    with col2:
        cat2_bar = st.selectbox("Select the grouping column (Color)", cat2_bar_options, key=f"adv_cat2_bar_{selected_plot}")

    if cat1_bar != "Select a column" and cat2_bar != "Select a column":
        try:
            counts_df = data.groupby([cat1_bar, cat2_bar]).size().reset_index(name='Count')
            fig_grouped_bar = px.bar(counts_df, x=cat1_bar, y='Count', color=cat2_bar,
                                     barmode='group',
                                     title=f"Grouped Bar Chart of {cat1_bar} by {cat2_bar}",
                                     text_auto=True)
            fig_grouped_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_grouped_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {e}")

def display_visualizations(data):
    st.subheader("Visualization Gallery")
    
    globally_filtered_data = data 
    with st.expander("Expand to Visualize the Data"):
        globally_filtered_data = render_global_filter_widgets(data)
        
        basic_tab, advanced_tab = st.tabs(["Basic Visualization", "Advanced Visualization"])

        with basic_tab:
            display_basic_visualizations(globally_filtered_data)

        with advanced_tab:
            display_advanced_visualizations(globally_filtered_data)
    
    return globally_filtered_data


def display_basic_visualizations(data):    
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    recommended_plots = recommend_visualizations(data)
    if not recommended_plots:
        st.warning("No visualizations could be recommended for the current data types.")
        return

    selected_plots = st.multiselect("**Select one or more visualizations to generate**", recommended_plots, key="basic_plots_select")

    plot_functions = {
        "Bar Chart": (_render_bar_chart, {"categorical_columns": categorical_columns}),
        "Histogram": (_render_histogram, {"numeric_columns": numeric_columns}),
        "Box Plot": (_render_box_plot, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Scatter Plot": (_render_scatter_plot, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Donut Chart": (_render_donut_chart, {"categorical_columns": categorical_columns}),
        "Heatmap": (_render_heatmap, {"numeric_columns": numeric_columns}),
        "Line Chart": (_render_line_chart, {"numeric_columns": numeric_columns})
    }

    for plot_type in selected_plots:
        if plot_type in plot_functions:
            with st.container(border=True):
                st.write(f"**{plot_type}**")
                locally_filtered_df = render_local_filter_widgets(data, chart_key=plot_type)
                if locally_filtered_df.empty:
                    st.warning("No data matches the current filter settings.")
                    continue
                
                func, kwargs = plot_functions[plot_type]
                func(locally_filtered_df, plot_type, **kwargs)

def recommend_visualizations(df):
    recommended_vis = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 1:
        recommended_vis.extend(["Scatter Plot", "Heatmap"])
    if len(categorical_cols) > 0:
        recommended_vis.append("Bar Chart")
    if len(numeric_cols) > 0:
        recommended_vis.append("Histogram")
        recommended_vis.append("Box Plot")
    if len(categorical_cols) > 0:
        recommended_vis.append("Donut Chart")
    if any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns):
        if "Line Chart" not in recommended_vis:
            recommended_vis.append("Line Chart")
    return list(set(recommended_vis))

def display_advanced_visualizations(data):
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    plot_types = [
        "Aggregated Bar Chart", "Grouped Line Chart", "Violin Plot (Categorical vs. Numeric)",
        "Enhanced Scatter Plot", "One-vs-Many Comparison (Individual Plots)",
        "Heatmap (Categorical vs. Categorical)", "Grouped Bar Chart (Categorical vs. Categorical )"
    ]
    selected_plots_list = st.multiselect("**Choose an advanced visualization to generate**", plot_types, key="advanced_plots_select")

    plot_functions = {
        "Aggregated Bar Chart": (_render_aggregated_bar_chart, {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}),
        "Grouped Line Chart": (_render_grouped_line_chart, {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}),
        "Violin Plot (Categorical vs. Numeric)": (_render_violin_plot, {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}),
        "Enhanced Scatter Plot": (_render_enhanced_scatter_plot, {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}),
        "One-vs-Many Comparison (Individual Plots)": (_render_one_vs_many_comparison, {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}),
        "Heatmap (Categorical vs. Categorical)": (_render_categorical_heatmap, {"categorical_cols": categorical_cols}),
        "Grouped Bar Chart (Categorical vs. Categorical )": (_render_grouped_bar_chart, {"categorical_cols": categorical_cols})
    }

    for selected_plot in selected_plots_list:
        if selected_plot in plot_functions:
            with st.container(border=True):
                chart_key = ''.join(e for e in selected_plot if e.isalnum())
                locally_filtered_df = render_local_filter_widgets(data, chart_key=chart_key)
                if locally_filtered_df.empty:
                    st.warning("No data matches the current filter settings.")
                    continue
                func, kwargs = plot_functions[selected_plot]
                func(locally_filtered_df, selected_plot, **kwargs)
