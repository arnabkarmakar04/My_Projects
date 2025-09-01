import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_ai_response
from utils.code_validator import CodeValidator

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
                        (filtered_data[col].dt.date >= start_date) & (filtered_data[col].dt.date <= end_date)]
            else:
                filter_container.info(f"Column '{col}' contains dates from a single day and cannot be filtered with a date range.")
    return filtered_data

def _render_bar_chart(data, plot_type, categorical_columns, numeric_columns):
    st.markdown("### Bar Chart Settings")
    x_col = st.selectbox("X-axis", [""] + numeric_columns + categorical_columns, key=f"bar_x_{plot_type}")
    if not x_col or x_col not in data.columns:
        return

    st.markdown("#### Optional Settings")
    col1, col2 = st.columns(2)
    with col1:
        y_col = st.selectbox("Y-axis (optional)", ["None"] + numeric_columns + categorical_columns, key=f"bar_y_{plot_type}")
        y_col = None if y_col == "None" or y_col not in data.columns else y_col
        if x_col == y_col:
            st.warning("Same columns are selected for X and Y. Please choose different columns.")
            return
    with col2:
        color_col = st.selectbox(
            "Color by", ["None"] + categorical_columns, key=f"bar_color_{plot_type}",
            help="Use categorical columns for coloring bars."
        )
        color_col = None if color_col == "None" or color_col not in data.columns else color_col

    st.markdown("#### Aggregation")
    col3, col4 = st.columns(2)
    with col3:
        agg_func = st.selectbox(
            "Aggregation (optional)",
            ["None", "sum", "mean", "median", "min", "max"],
            key=f"bar_agg_{plot_type}",
            help="This shows counts if no numeric aggregation is chosen."
        )
    with col4:
        agg_y_col = None
        if agg_func != "None":
            agg_y_col = st.selectbox(
                "Numeric column for aggregation",
                ["None"] + numeric_columns,
                key=f"bar_agg_y_{plot_type}",
                help="This column is used for aggregation. If selected, Y-axis selection is ignored."
            )
    agg_y_col = None if agg_y_col == "None" or agg_y_col not in data.columns else agg_y_col
    st.markdown("#### Faceting / Animation (Choose One)")
    use_animation = st.checkbox("Use Animation?", key=f"bar_use_animation_{plot_type}")

    facet_col = None
    facet_col_wrap = None
    animation_col = None

    if use_animation:
        safe_anim_cols = list(dict.fromkeys([x_col] + categorical_columns))
        animation_col = st.selectbox(
            "Animation frame column", ["None"] + safe_anim_cols, key=f"bar_animate_col_{plot_type}",
            help="Use categorical or datetime columns for animation."
        )
        animation_col = None if animation_col == "None" or animation_col not in data.columns else animation_col
    else:
        facet_col = st.selectbox("Facet by", ["None"] + categorical_columns, key=f"bar_facet_{plot_type}")
        facet_col = None if facet_col == "None" or facet_col not in data.columns else facet_col
        if facet_col:
            facet_col_wrap = st.number_input(
                "Facet wrap (0 for no wrap)", min_value=0, value=3, step=1, key=f"bar_facet_wrap_{plot_type}"
            )
            facet_col_wrap = facet_col_wrap if facet_col_wrap > 0 else None

    if animation_col and color_col and animation_col == color_col:
        st.warning("Animation and Color columns cannot be the same.")
        return
    if x_col and agg_y_col and x_col == agg_y_col:
        st.warning("X-axis and Aggregation columns cannot be the same.")
        return
    
    df = data.copy()
    groupby_cols = [x_col]
    if color_col:
        groupby_cols.append(color_col)
    if facet_col:
        groupby_cols.append(facet_col)
    if animation_col:
        groupby_cols.append(animation_col)
    groupby_cols = [c for c in dict.fromkeys(groupby_cols) if c in df.columns]

    if not y_col and agg_func == "None":
        df = df.groupby(groupby_cols, dropna=False).size().reset_index(name="Count")
        y_arg = "Count"
        title = f"Count of {x_col}"
        show_text = True
    elif agg_func != "None":
        if not agg_y_col:
            st.info("Please select numeric column for aggregation. Using count instead.")
            df = df.groupby(groupby_cols, dropna=False).size().reset_index(name="Count")
            y_arg = "Count"
        else:
            df = df.groupby(groupby_cols, dropna=False)[agg_y_col].agg(agg_func).reset_index()
            y_arg = agg_y_col
        title = f"{agg_func.capitalize()} of {y_arg} by {x_col}"
        show_text = True
    else:
        y_arg = y_col
        title = f"{y_col} by {x_col}"
        show_text = False
    if color_col:
        title += f", colored by {color_col}"
    if animation_col and animation_col in df.columns:
        df = df.sort_values(by=animation_col)
    fig = px.bar(
        df,
        x=x_col,
        y=y_arg,
        color=color_col,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        animation_frame=animation_col if animation_col in df.columns else None,
        barmode="group",
        title=title,
        text=y_arg if show_text else None,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    if show_text:
        fig.update_traces(texttemplate="%{text}", textposition="outside", marker_line_width=0)
    else:
        fig.update_traces(marker_line_width=0)
    fig.update_layout(showlegend=bool(color_col))
    st.plotly_chart(fig, use_container_width=True)

def _render_histogram(data, plot_type, numeric_columns, categorical_columns):
    st.markdown("### Histogram Settings")
    hist_col = st.selectbox(
        "Numeric column (X-axis)",
        [""] + numeric_columns,
        key=f"hist_{plot_type}"
    )
    if not hist_col:
        return
    st.markdown("#### Optional Aesthetics")
    col1, col2 = st.columns(2)
    with col1:
        color_col = st.selectbox(
            "Color by",
            ["None"] + categorical_columns,
            key=f"hist_color_{plot_type}"
        )
        color_col = None if color_col == "None" else color_col
    with col2:
        agg_func = st.selectbox(
            "Aggregation (optional)",
            ["None", "sum", "avg", "min", "max"],
            key=f"hist_agg_{plot_type}",
            help="Count is default aggregation if no aggregation method is selected."
        )
    y_col = None
    if agg_func != "None":
        y_col = st.selectbox(
            "Aggregate on (numeric)",
            [""] + numeric_columns,
            key=f"hist_y_{plot_type}"
        )
        if not y_col:
            y_col = None
    st.markdown("#### Faceting")
    facet_col = st.selectbox(
        "Facet by",
        ["None"] + categorical_columns,
        key=f"hist_facet_{plot_type}"
    )
    facet_col = None if facet_col == "None" else facet_col
    facet_col_wrap = None
    if facet_col:
        facet_col_wrap = st.number_input(
            "Facet wrap (0 for no wrap)", min_value=0, value=3, step=1, key=f"bar_facet_wrap_{plot_type}"
        )
        facet_col_wrap = facet_col_wrap if facet_col_wrap > 0 else None
    if hist_col and y_col and hist_col == y_col:
        st.warning("Same columns are selected for X and Y. Please choose different columns.")
        return
    df = data.copy()
    histfunc = "count" if agg_func == "None" else agg_func
    fig = px.histogram(
        df,
        x=hist_col,
        y=y_col,
        color=color_col,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        histfunc=histfunc,
        title=f"Histogram of {hist_col}",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

def _render_box_plot(data, plot_type, numeric_columns, categorical_columns):
    st.markdown("### Box Plot Settings")
    y_col_box = st.selectbox(
        "Y-axis (numeric column)",
        [""] + numeric_columns,
        key=f"boxplot_y_{plot_type}"
    )
    if not y_col_box:
        return
    st.markdown("#### Optional Aesthetics")
    col1, col2 = st.columns(2)
    with col1:
        x_col_box = st.selectbox(
            "Group by (X-axis)",
            ["None"] + categorical_columns,
            key=f"boxplot_x_{plot_type}"
        )
        x_col_box = None if x_col_box == "None" else x_col_box
    with col2:
        color_col_box = st.selectbox(
            "Color by",
            ["None"] + categorical_columns,
            key=f"boxplot_color_{plot_type}"
        )
        color_col_box = None if color_col_box == "None" else color_col_box
    st.markdown("#### Faceting")
    facet_col = st.selectbox(
        "Facet by",
        ["None"] + categorical_columns,
        key=f"boxplot_facet_{plot_type}"
    )
    facet_col = None if facet_col == "None" else facet_col
    facet_col_wrap = None
    if facet_col:
        facet_col_wrap = st.number_input(
            "Facet wrap (0 for no wrap)", min_value=0, value=3, step=1, key=f"bar_facet_wrap_{plot_type}"
        )
        facet_col_wrap = facet_col_wrap if facet_col_wrap > 0 else None
    df= data.copy()
    st.markdown("#### Points")
    points_opt = st.selectbox(
        "Show points",
        ["outliers", "all", "suspectedoutliers", False],
        key=f"boxplot_points_{plot_type}"
    )
    fig_box = px.box(
        df,
        x=x_col_box,
        y=y_col_box,
        color=color_col_box,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        points=points_opt,
        title=f"Box Plot of {y_col_box}" + (f" grouped by {x_col_box}" if x_col_box else ""),
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    if x_col_box and color_col_box == x_col_box:
        fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

def _render_scatter_plot(data, plot_type, numeric_columns, categorical_columns, animatable_columns=None):
    st.markdown("### Scatter Plot Settings")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis", [""] + numeric_columns, key=f"x_{plot_type}")
    with col2:
        y_col_options = [col for col in numeric_columns if col != x_col]
        y_col = st.selectbox("Y-axis", [""] + y_col_options, key=f"y_{plot_type}")
    if not x_col or not y_col:
        return
    st.markdown("#### Optional Aesthetics")
    col3, col4, col5 = st.columns(3)
    with col3:
        color_col = st.selectbox("Color by", ["None"] + categorical_columns, key=f"color_{plot_type}")
    with col4:
        size_col = st.selectbox("Size by", ["None"] + numeric_columns, key=f"size_{plot_type}")
    with col5:
        symbol_col = st.selectbox("Symbol by", ["None"] + categorical_columns, key=f"symbol_{plot_type}")
    trendline_opt = st.selectbox("Trendline", ["None", "ols", "lowess"], key=f"trendline_{plot_type}")
    st.markdown("#### Faceting")
    facet_col = st.selectbox("Facet by", ["None"] + categorical_columns, key=f"facet_{plot_type}")
    facet_col = None if facet_col == "None" else facet_col
    facet_col_wrap = None
    if facet_col:
        facet_col_wrap = st.number_input("Facet wrap (0 for no wrap)", min_value=0, value=3, step=1, key=f"facet_wrap_{plot_type}")
        facet_col_wrap = facet_col_wrap if facet_col_wrap > 0 else None
    st.markdown("#### Animation")
    animate_checkbox = st.checkbox("Animate chart?", key=f"animate_check_{plot_type}")
    animation_col = None
    if animate_checkbox:
        animation_col = st.selectbox("Animation frame column", ["None"] + animatable_columns, key=f"animate_col_{plot_type}")
        animation_col = None if animation_col == "None" else animation_col
    if facet_col!= None and animation_col != None and facet_col == animation_col:
        st.warning("Facet column and Animation column cannot be the same.")
        return
    df_for_plot = data.copy()
    if animation_col and animation_col in df_for_plot.columns:
        df_for_plot = df_for_plot.sort_values(by=animation_col)
    fig = px.scatter(
        df_for_plot,
        x=x_col,
        y=y_col,
        color=None if color_col == "None" else color_col,
        size=None if size_col == "None" else size_col,
        symbol=None if symbol_col == "None" else symbol_col,
        trendline=None if trendline_opt == "None" else trendline_opt,
        title=f"Scatter Plot: {y_col} vs {x_col}",
        hover_data=data.columns,
        animation_frame=animation_col,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap, 
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(template="plotly_white")
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig, use_container_width=True)

def _render_donut_chart(data, plot_type, categorical_columns, numeric_columns):
    st.markdown("### Donut Chart Settings")
    name_col = st.selectbox(
        "Names (categorical)",
        [""] + categorical_columns,
        key=f"donut_name_{plot_type}"
    )
    if not name_col:
        return
    st.markdown("#### Optional Aesthetics")
    col1, col2, col3= st.columns(3)
    with col1:
        value_col = st.selectbox(
            "Values (numeric, optional)",
            [""] + numeric_columns,
            key=f"donut_val_{plot_type}"
        )
    with col2:
        color_col = st.selectbox(
            "Color by",
            ["None"] + categorical_columns,
            key=f"donut_color_{plot_type}"
        )
    with col3:
        agg_func = st.selectbox(
            "Aggregation",
            ["None", "sum", "mean", "median", "min", "max"],
            key=f"donut_agg_{plot_type}",
            help="Count is default aggregation if no aggregation method is selected."
        )
    col4 = st.columns(1)[0]
    with col4:
        st.markdown("#### Faceting")
        facet_col = st.selectbox(
            "Facet by (optional)",
            ["None"] + categorical_columns,
            key=f"donut_facet_{plot_type}"
        )
    facet_wrap = None
    if facet_col != "None":
        with st.container():
            facet_wrap = st.number_input(
                "Facet Wrap (0 for no wrap)",
                min_value=0,
                value=3,
                step=1,
                key=f"donut_facet_wrap_{plot_type}"
            )
    if color_col!='None' and facet_col !='None' and color_col == facet_col:
        st.warning("Color column and facet column cannot be the same. Please select different columns.")
        return
    if name_col and facet_col and name_col == facet_col:
        st.warning("Name column and facet column cannot be the same. Please select different columns.")
        return
    df = data.copy()
    if not value_col or agg_func == "None":
        if color_col != "None" and color_col != name_col:
            df = df.groupby([name_col, color_col] + ([] if facet_col == "None" else [facet_col])).size().reset_index(name="Count")
            df["label"] = df[name_col].astype(str) + " | " + df[color_col].astype(str)
            fig = px.pie(
                df,
                names="label",
                values="Count",
                color=color_col,
                facet_col=None if facet_col == "None" else facet_col,
                facet_col_wrap=facet_wrap,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"Donut Chart of {name_col} by {color_col} (Count)"
            )
        else:
            df = df.groupby([name_col] + ([] if facet_col == "None" else [facet_col])).size().reset_index(name="Count")
            fig = px.pie(
                df,
                names=name_col,
                values="Count",
                facet_col=None if facet_col == "None" else facet_col,
                facet_col_wrap=facet_wrap,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"Donut Chart of {name_col} (Count)"
            )
    else:
        pie_agg = agg_func
        if color_col != "None" and color_col != name_col:
            df = df.groupby([name_col, color_col] + ([] if facet_col == "None" else [facet_col]))[value_col].agg(pie_agg).reset_index()
            df["label"] = df[name_col].astype(str) + " | " + df[color_col].astype(str)
            fig = px.pie(
                df,
                names="label",
                values=value_col,
                color=color_col,
                facet_col=None if facet_col == "None" else facet_col,
                facet_col_wrap=facet_wrap,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"Donut Chart of {value_col} by {name_col} and {color_col} ({pie_agg})"
            )
        else:
            df = df.groupby([name_col] + ([] if facet_col == "None" else [facet_col]))[value_col].agg(pie_agg).reset_index()
            fig = px.pie(
                df,
                names=name_col,
                values=value_col,
                facet_col=None if facet_col == "None" else facet_col,
                facet_col_wrap=facet_wrap,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"Donut Chart of {value_col} by {name_col} ({pie_agg})"
            )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(data, plot_type, numeric_columns):
    st.write("This heatmap shows the correlation between all numeric columns in the data.")
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Plasma",
            title=f"Correlation Heatmap - {plot_type}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Heatmap requires at least two numeric columns.")

def _render_line_chart(data, plot_type, categorical_columns, numeric_columns, datetime_cols):
    all_columns = list(dict.fromkeys(categorical_columns + numeric_columns + datetime_cols))

    st.markdown("### Line Chart Settings")
    x_col = st.selectbox("X-axis", [""] + all_columns, key=f"line_x_{plot_type}")
    if not x_col or x_col not in data.columns:
        return

    st.markdown("#### Optional Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        y_col = st.selectbox(
            "Y-axis (numeric)", [""] + numeric_columns, key=f"line_y_{plot_type}"
        )
    with col2:
        color_col = st.selectbox(
            "Color by", ["None"] + categorical_columns, key=f"line_color_{plot_type}",
            help="Use categorical columns for grouping lines."
        )
        color_col = None if color_col == "None" or color_col not in data.columns else color_col
    with col3:
        agg_func = st.selectbox(
            "Aggregation (optional)",
            ["None", "count", "sum", "mean", "median", "min", "max"],
            key=f"line_agg_{plot_type}",
            help="Applies aggregation over X-axis values if multiple Y values exist per X."
        )

    st.markdown("#### Faceting / Animation (Choose One)")
    use_animation = st.checkbox("Use Animation?", key=f"line_use_animation_{plot_type}")

    facet_col = None
    facet_col_wrap = None
    animation_col = None

    if use_animation:
        safe_anim_cols = list(dict.fromkeys([x_col] + datetime_cols + categorical_columns))
        animation_col = st.selectbox(
            "Animation frame column", ["None"] + safe_anim_cols, key=f"line_animate_col_{plot_type}",
            help="Use datetime or categorical columns for animation."
        )
        animation_col = None if animation_col == "None" or animation_col not in data.columns else animation_col
    else:
        facet_col = st.selectbox("Facet by", ["None"] + categorical_columns, key=f"line_facet_{plot_type}")
        facet_col = None if facet_col == "None" or facet_col not in data.columns else facet_col
        if facet_col:
            facet_col_wrap = st.number_input(
                "Facet wrap (0 for no wrap)", min_value=0, value=3, step=1, key=f"line_facet_wrap_{plot_type}"
            )
            facet_col_wrap = facet_col_wrap if facet_col_wrap > 0 else None

    df = data.copy()

    groupby_cols = [x_col]
    if color_col:
        groupby_cols.append(color_col)
    if facet_col:
        groupby_cols.append(facet_col)
    if animation_col:
        groupby_cols.append(animation_col)
    groupby_cols = [c for c in dict.fromkeys(groupby_cols) if c in df.columns]

    auto_count = False
    if not y_col:
        y_col = "Count"
        df[y_col] = 1
        auto_count = True

    if agg_func == "None" and not auto_count:
        agg_df = df.copy()
        title = f"{y_col} over {x_col}"
    else:
        if groupby_cols:
            if auto_count:
                agg_df = df.groupby(groupby_cols, dropna=False).size().reset_index(name=y_col)
            else:
                agg_df = df.groupby(groupby_cols, dropna=False)[y_col].agg(agg_func).reset_index()
        else:
            if auto_count:
                agg_df = df.groupby(x_col, dropna=False).size().reset_index(name=y_col)
            else:
                agg_df = df[[x_col, y_col]].groupby(x_col, dropna=False)[y_col].agg(agg_func).reset_index()

        if auto_count:
            title = f"Count of {x_col}"
        else:
            title = f"{agg_func.capitalize()} of {y_col} over {x_col}"
        if color_col:
            title += f" by {color_col}"

    if animation_col and animation_col in agg_df.columns:
        agg_df = agg_df.sort_values(by=animation_col)

    fig = px.line(
        agg_df,
        x=x_col,
        y=y_col,
        color=color_col,
        facet_col=facet_col if facet_col in agg_df.columns else None,
        facet_col_wrap=facet_col_wrap,
        animation_frame=animation_col if animation_col in agg_df.columns else None,
        title=title,
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    y_min = agg_df[y_col].min()
    y_max = agg_df[y_col].max()
    fig.update_yaxes(range=[y_min * 0.95, y_max * 1.05])
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig,use_container_width=True)

def _render_violin_plot(data, selected_plot, numeric_columns, categorical_columns):
    st.markdown("### Violin Plot Settings")
    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("X-axis (Categorical)", [""] + categorical_columns, key=f"adv_cat_col_{selected_plot}")
    with col2:
        num_col = st.selectbox("Y-axis (Numeric)", [" "] + numeric_columns, key=f"adv_num_col_{selected_plot}")
    if cat_col != " " and num_col != " ":
        st.markdown("#### Optional Aesthetics")
        col3, col4 = st.columns(2)
        with col3:
            color_col = st.selectbox("Color by", ["None"] + categorical_columns, key=f"adv_color_col_{selected_plot}")
        with col4:
            points_val = st.selectbox(
                "Points",
                ["outliers", "suspectedoutliers", "None", "all"],
                index=0,
                key=f"adv_points_col_{selected_plot}"
            )
        final_color = None if color_col == "None" else color_col
        final_points = False if points_val == "None" else points_val
        df= data.copy()
        fig = px.violin(
            df,
            x=cat_col,
            y=num_col,
            color=final_color,
            title=f"Distribution of '{num_col}' by '{cat_col}'",
            box=True,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            points=final_points
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def _render_3D_scatter_plot(data, plot_type, numeric_columns, categorical_columns):
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X-axis", [""] + numeric_columns, key=f"scatter_x_{plot_type}")
    with col2:
        y_col_options = [""] + [col for col in numeric_columns if col != x_col]
        y_col = st.selectbox("Y-axis", y_col_options, key=f"scatter_y_{plot_type}")
    with col3:
        z_col_options = [""] + [col for col in numeric_columns if col not in [x_col, y_col]]
        z_col = st.selectbox("Z-axis", z_col_options, key=f"scatter_z_{plot_type}")

    if not x_col or not y_col or not z_col:
        return

    st.markdown("#### Optional Settings")
    col4, col5 = st.columns(2)
    with col4:
        color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key=f"scatter_color_{plot_type}")
    with col5:
        size_col = st.selectbox("Size by (optional)", ["None"] + numeric_columns, key=f"scatter_size_{plot_type}")

    if len({x_col, y_col, z_col}) < 3:
        st.warning("X, Y, and Z axes must all be different columns.")
        return

    st.markdown("#### Aggregation")
    agg_col1, agg_col2 = st.columns(2)
    with agg_col1:
        agg_func = st.selectbox(
            "Aggregation Function",
            ["None", "count", "sum", "mean", "median", "min", "max"],
            key=f"scatter_agg_func_{plot_type}"
        )
    with agg_col2:
        if agg_func != "None":
            agg_target_col = st.selectbox(
                "Aggregation Column",
                ["None"] + numeric_columns,
                key=f"scatter_agg_target_{plot_type}"
            )

    df = data.copy()
    df_for_plot = df

    if agg_func != "None" and agg_target_col != "None":
        groupby_cols = [x_col, y_col, z_col]
        if color_col != "None":
            groupby_cols.append(color_col)

        df_agg = df.groupby(groupby_cols, as_index=False)[agg_target_col].agg(agg_func)
        df_for_plot = df_agg
        
        if agg_target_col not in df_for_plot.columns:
            df_for_plot = df_for_plot.rename(columns={agg_target_col: f"{agg_func.capitalize()} of {agg_target_col}"})

    fig = px.scatter_3d(
        df_for_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color=None if color_col == "None" else color_col,
        size=None if size_col == "None" else size_col,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
        hover_data=df_for_plot.columns
    )
    fig.update_layout(template="plotly_white")
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig, use_container_width=True)

def _render_sunburst_chart(data, plot_type, categorical_columns, numeric_columns):
    st.markdown("### Sunburst Chart Settings")
    if not categorical_columns:
        st.warning("No categorical columns available for Sunburst chart.")
        return
    st.markdown("#### Hierarchy Selection")
    level_cols = st.multiselect(
        "Select hierarchy columns (at least one required)",
        categorical_columns,
        key=f"sunburst_levels_{plot_type}"
    )
    if not level_cols:
        return
    st.markdown("#### Optional Asthetics")
    value_col = st.selectbox(
        "Value (numeric, optional)",
        ["None"] + numeric_columns,
        key=f"sunburst_value_{plot_type}"
    )
    value_col = None if value_col == "None" else value_col
    agg_func = "count" if not value_col else st.selectbox(
        "Aggregation",
        ["count", "sum", "mean", "median", "min", "max"],
        key=f"sunburst_agg_{plot_type}"
    )
    df = data.copy()
    if value_col:
        agg_df = df.groupby(level_cols, dropna=False)[value_col].agg(agg_func).reset_index()
    else:
        agg_df = df.groupby(level_cols, dropna=False).size().reset_index(name="count")
        value_col = "count"
    fig = px.sunburst(
        agg_df,
        path=level_cols,
        values=value_col,
        title=("Sunburst Chart" if value_col == "count" else f"Sunburst Chart ({agg_func} of {value_col})"),
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def _render_3D_line_chart(data, plot_type, numeric_columns, categorical_columns, datetime_cols):
    st.markdown("### 3D Line Chart Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox(
            "X-axis", [""] + numeric_columns + categorical_columns + datetime_cols,
            key=f"line3d_x_{plot_type}"
        )
    with col2:
        y_col = st.selectbox(
            "Y-axis", [""] + numeric_columns,
            key=f"line3d_y_{plot_type}"
        )
    with col3:
        z_col = st.selectbox(
            "Z-axis", [""] + numeric_columns + categorical_columns + datetime_cols,
            key=f"line3d_z_{plot_type}"
        )

    if not x_col or not y_col or not z_col:
        return

    if len({x_col, y_col, z_col}) < 3:
        st.warning("X, Y, and Z axes must all be different columns.")
        return

    st.markdown("#### Optional Aesthetics")
    color_col = st.selectbox(
        "Color by (optional)", ["None"] + categorical_columns,
        key=f"line3d_color_{plot_type}"
    )
    color_col = None if color_col == "None" or color_col not in data.columns else color_col
    st.markdown("#### Aggregation")
    col1, col2 = st.columns(2)
    with col1:
        agg_func = st.selectbox(
            "Aggregation Function (optional)",
            ["None", "count", "sum", "mean", "median", "min", "max"],
            key=f"line3d_agg_func_{plot_type}"
        )
    with col2:
        if agg_func != "None":
            agg_target_col = st.selectbox(
                "Aggregation Target",
                ["None"] + numeric_columns,
                key=f"line3d_agg_target_{plot_type}"
            )

    df = data.copy()
    if agg_func != "None" and agg_target_col != "None":
        groupby_cols = [x_col, z_col]
        if color_col:
            groupby_cols.append(color_col)
        
        agg_df = df.groupby(groupby_cols)[agg_target_col].agg(agg_func).reset_index()
        title = f"{agg_func.capitalize()} of {agg_target_col} over {x_col} & {z_col}"
        
        fig = px.line_3d(
            agg_df,
            x=x_col,
            y=agg_target_col,
            z=z_col,
            color=color_col,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
    else:
        title = f"{y_col} over {x_col} & {z_col}"
        fig = px.line_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )

    fig.update_traces(line=dict(width=4))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

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
        
        # Determine the number of columns to display the plots
        num_columns = 2
        cols = st.columns(num_columns)
        
        color_arg_sm = None if color_col_sm == "Select a column" else color_col_sm
        
        for i, other_col in enumerate(other_cols):
            with cols[i % num_columns]:
                try:
                    fig = px.scatter(data, x=main_col, y=other_col, color=color_arg_sm,
                                     title=f"'{main_col}' vs. '{other_col}'")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate plot for '{other_col}'. Error: {e}")

def _render_treemap(data, plot_type, categorical_columns, numeric_columns):
    st.markdown("### Treemap Settings")    
    path_cols = st.multiselect("Hierarchy (categorical columns)", categorical_columns, key=f"treemap_path_{plot_type}")
    if not path_cols:
        st.warning("Select at least one categorical column for hierarchy.")
        return
    value_col = st.selectbox("Values (numeric column)", ["None"] + numeric_columns, key=f"treemap_value_{plot_type}")
    value_col = None if value_col == "None" else value_col
    agg_func = st.selectbox("Aggregation (optional)", ["None", "sum", "mean", "median", "min", "max"], key=f"treemap_agg_{plot_type}")
    df = data.copy()
    if value_col and agg_func != "None":
        df = df.groupby(path_cols, dropna=False)[value_col].agg(agg_func).reset_index()
    title = f"Treemap of {value_col} by {' > '.join(path_cols)}" if value_col else f"Treemap by {' > '.join(path_cols)}"
    fig = px.treemap(
        df,
        path=path_cols,
        values=value_col,
        title=title,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_area_chart(data, plot_type, numeric_columns, categorical_columns, datetime_cols):
    st.markdown("### Area Chart Settings")    
    all_x_cols = numeric_columns + datetime_cols
    x_col = st.selectbox("X-axis", [""] + all_x_cols, key=f"area_x_{plot_type}")
    if not x_col:
        return
    y_col = st.selectbox("Y-axis (numeric)", [""] + numeric_columns, key=f"area_y_{plot_type}")
    color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key=f"area_color_{plot_type}")
    color_col = None if color_col == "None" else color_col
    facet_col = st.selectbox("Facet by (optional)", ["None"] + categorical_columns, key=f"area_facet_{plot_type}")
    facet_col = None if facet_col == "None" else facet_col
    agg_func = st.selectbox("Aggregation (optional)", ["None", "sum", "mean", "median", "min", "max"], key=f"area_agg_{plot_type}")
    df = data.copy()
    groupby_cols = [x_col]
    if color_col:
        groupby_cols.append(color_col)
    if facet_col:
        groupby_cols.append(facet_col)
    if agg_func != "None" and y_col:
        df = df.groupby(groupby_cols, dropna=False)[y_col].agg(agg_func).reset_index()
    title = f"{agg_func.capitalize()} of {y_col} over {x_col}" if agg_func != "None" and y_col else f"{y_col} over {x_col}"
    fig = px.area(
        df,
        x=x_col,
        y=y_col if y_col else None,
        color=color_col,
        facet_col=facet_col,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_parallel_coordinates_chart(data, plot_type, numeric_columns):
    st.markdown("### Parallel Coordinates Chart Settings")    
    selected_numeric_cols = st.multiselect("Select numeric columns", numeric_columns, key=f"parcoords_numeric_{plot_type}")
    if len(selected_numeric_cols) < 2:
        st.warning("Select at least 2 numeric columns for parallel coordinates chart.")
        return
    color_col = st.selectbox("Color by (optional)", ["None"]+ numeric_columns, key=f"parcoords_color_{plot_type}")
    color_col = None if color_col == "None" else color_col
    df = data.copy()
    fig = px.parallel_coordinates(
        df,
        dimensions=selected_numeric_cols,
        color=color_col if color_col else None,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

def display_visualizations(data):
    st.subheader("Visualization Gallery")    
    globally_filtered_data = data 
    with st.expander("Expand to Visualize the Data"):
        globally_filtered_data = render_global_filter_widgets(data)
        st.markdown("💬 Ask Your Data")
        with st.form(key='viz_query_form', clear_on_submit=True):
            viz_query_text = st.text_input("Describe the plot you want to create:", key="viz_query_input")
            viz_submitted = st.form_submit_button("Generate Plot")
        if 'last_viz_result' not in st.session_state:
            st.session_state.last_viz_result = None
        if viz_submitted and viz_query_text:
            with st.spinner("🤖 Generating visualization..."):
                df = globally_filtered_data.copy()
                prompt = f"""
                You are an expert in Python data visualization using the Plotly Express library.
                A user has a pandas DataFrame named `df`. The columns of the DataFrame are: {', '.join(df.columns)}.
                The user wants to create a plot and has given this instruction: "{viz_query_text}"

                Your task is to generate a Python code snippet using Plotly Express to create the requested visualization.

                **RULES:**
                - Respond with ONLY the Python code for the plot. Do not include explanations, comments, or the word "python".
                - The final plot object MUST be assigned to a variable named `result`.
                - Use the `px` alias for `plotly.express`.
                - DO NOT use `print()` or `fig.show()`.
                """
                generated_code = get_ai_response(prompt)
                if generated_code:
                    is_safe, error_message = CodeValidator.validate(generated_code)
                    if is_safe:
                        try:
                            local_vars = {'df': df, 'px': px}
                            exec(generated_code, {}, local_vars)
                            result = local_vars.get('result', None)
                            st.session_state.last_viz_result = result
                        except Exception as e:
                            st.session_state.last_viz_result = f"An error occurred while generating the plot: {e}"
                    else:
                        st.session_state.last_viz_result = f"Execution stopped for security reasons: {error_message}"
                else:
                    st.session_state.last_viz_result = "Warning: The AI model could not generate a response for the visualization."
        if st.session_state.last_viz_result:
            if isinstance(st.session_state.last_viz_result, str):
                st.error(st.session_state.last_viz_result)
            else:
                st.plotly_chart(st.session_state.last_viz_result, use_container_width=True)
        basic_tab, advanced_tab = st.tabs(["Basic Visualization", "Advanced Visualization"])
        with basic_tab:
            display_basic_visualizations(globally_filtered_data)
        with advanced_tab:
            display_advanced_visualizations(globally_filtered_data)
    return globally_filtered_data

def display_basic_visualizations(data):
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    animatable_columns = numeric_columns + datetime_cols + categorical_columns
    plot_types = [
        "Bar Chart", "Histogram", "Box Plot", "Scatter Plot",
        "Donut Chart", "Violin Plot", "Line Chart", "Heatmap"
    ]
    plot_functions = {
        "Bar Chart": (_render_bar_chart, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Histogram": (_render_histogram, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Box Plot": (_render_box_plot, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Scatter Plot": (_render_scatter_plot, {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "animatable_columns": animatable_columns
        }),
        "Donut Chart": (_render_donut_chart, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Violin Plot": (_render_violin_plot, {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}),
        "Heatmap": (_render_heatmap, {"numeric_columns": numeric_columns}),
        "Line Chart": (_render_line_chart, {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_cols": datetime_cols,
        })
    }
    selected_plots_list = st.multiselect(
        "Select one or more plot types to generate",
        plot_types,
        key="basic_plots_select"
    )
    for selected_plot in selected_plots_list:
        if selected_plot in plot_functions:
            with st.container(border=True):
                plot_func, params = plot_functions[selected_plot]
                plot_func(data, selected_plot, **params)
           
def display_advanced_visualizations(data):
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    plot_types = [
        "3D Scatter Plot",
        "Sunburst Chart",
        "3D Line Chart",
        "One-vs-Many Comparison (Individual Plots)",
        "Treemap",
        "Area Chart",
        "Parallel Coordinates Chart"
    ]
    selected_plots_list = st.multiselect(
        "**Choose an advanced visualization to generate**",
        plot_types,
        key="advanced_plots_select"
    )
    plot_functions = {
        "3D Scatter Plot": (
            _render_3D_scatter_plot,
            {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}
        ),
        "Sunburst Chart": (
            _render_sunburst_chart,
            {"categorical_columns": categorical_columns, "numeric_columns": numeric_columns}
        ),
        "3D Line Chart": (
            _render_3D_line_chart,
            {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "datetime_cols": datetime_cols}
        ),
        "One-vs-Many Comparison (Individual Plots)": (
            _render_one_vs_many_comparison,
            {"numeric_cols": numeric_columns, "categorical_cols": categorical_columns}
        ),
        "Treemap": (
            _render_treemap,
            {"categorical_columns": categorical_columns, "numeric_columns": numeric_columns}
        ),
        "Area Chart": (
            _render_area_chart,
            {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "datetime_cols": datetime_cols}
        ),
        "Parallel Coordinates Chart": (
            _render_parallel_coordinates_chart,
            {"numeric_columns": numeric_columns}
        )
    }

    for selected_plot in selected_plots_list:
        if selected_plot in plot_functions:
            with st.container(border=True):
                func, kwargs = plot_functions[selected_plot]
                func(data, selected_plot, **kwargs)
