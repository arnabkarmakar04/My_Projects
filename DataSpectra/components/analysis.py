import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc

def display_advanced_analytics(data):
    st.subheader("Advanced Analytical Features")

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with st.expander("Select Analysis Methods"):
        test_size = st.slider(
            "Select Test Set Size", 
            min_value=0.1, 
            max_value=0.4, 
            value=0.3, 
            step=0.05,
            help="Select the proportion of the dataset to hold out for testing the models."
        )

        analysis_methods = st.multiselect(
            "Choose analysis methods to perform:",
            [
                "Linear Regression", 
                "Logistic Regression", 
                "Random Forest Classification", 
                "Decision Tree Classifier",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)",
                "K-Means Clustering"
            ]
        )

        if "Linear Regression" in analysis_methods:
            perform_linear_regression(data, numeric_columns, test_size)
        
        if "Logistic Regression" in analysis_methods:
            perform_logistic_regression(data, numeric_columns, categorical_columns, test_size)

        if "Random Forest Classification" in analysis_methods:
            perform_random_forest(data, numeric_columns, categorical_columns, test_size)
        
        if "Decision Tree Classifier" in analysis_methods:
            perform_decision_tree_classification(data, numeric_columns, categorical_columns, test_size)

        if "Support Vector Machine (SVM)" in analysis_methods:
            perform_svm_classification(data, numeric_columns, categorical_columns, test_size)

        if "K-Nearest Neighbors (KNN)" in analysis_methods:
            perform_knn_classification(data, numeric_columns, categorical_columns, test_size)

        if "K-Means Clustering" in analysis_methods:
            perform_kmeans_clustering(data, numeric_columns)

def perform_linear_regression(data, numeric_columns, test_size):
    st.write("---")
    st.write("#### Linear Regression Report")
    if len(numeric_columns) < 2:
        st.warning("Linear Regression requires at least two numeric columns.")
        return
    
    if 'lr_y' not in st.session_state:
        st.session_state.lr_y = "Select a column"
    if 'lr_x' not in st.session_state:
        st.session_state.lr_x = []

    y_col_selected = st.session_state.lr_y
    x_cols_selected = st.session_state.lr_x

    y_options = ["Select a column"] + [col for col in numeric_columns if col not in x_cols_selected]
    x_options = [col for col in numeric_columns if col != y_col_selected]

    if y_col_selected not in y_options:
        st.session_state.lr_y = "Select a column"
    st.session_state.lr_x = [col for col in x_cols_selected if col in x_options]

    col1, col2 = st.columns(2)
    with col1:
        x_cols = st.multiselect(
            "Select Independent variable(s) (X)",
            options=x_options,
            key="lr_x"
        )
    with col2:
        y_col = st.selectbox(
            "Select Dependent variable (Y)",
            options=y_options,
            key="lr_y"
        )

    if x_cols and y_col != "Select a column":
        # REMOVED: The .dropna() call. The component now assumes data is clean.
        temp_data = data[x_cols + [y_col]]
        
        if temp_data.shape[0] < 10:
            st.error("Not enough data to perform regression. A minimum of 10 rows is required.")
            return

        try:
            X = temp_data[x_cols]
            y = temp_data[y_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("##### Model Performance on Test Set")
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("R-squared (R²)", f"{r2:.3f}")
            m_col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
            m_col3.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
            
            st.markdown("---")

            if len(x_cols) == 1:
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.write("##### Regression Plot")
                    fig = px.scatter(temp_data, x=x_cols[0], y=y_col, trendline="ols",
                                         title=f'Trend: {y_col} vs {x_cols[0]}')
                    st.plotly_chart(fig, use_container_width=True)

                with plot_col2:
                    st.write("##### Residuals Plot")
                    residuals = y_test - y_pred
                    res_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                         title="Residuals vs. Predicted Values")
                    res_fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(res_fig, use_container_width=True)
                
                st.write(f"**Equation:** `{y_col} = {model.coef_[0]:.4f} * {x_cols[0]} + {model.intercept_:.4f}`")

            else:
                st.write("##### Individual Variable Plots")
                cols = st.columns(2) 
                for i, col_name in enumerate(x_cols):
                    with cols[i % 2]:
                        fig = px.scatter(temp_data, x=col_name, y=y_col,
                                             title=f'{y_col} vs {col_name}', trendline="ols", trendline_color_override="red")
                        st.plotly_chart(fig, use_container_width=True)
                
                equation_str = f"{model.intercept_:.4f}"
                for coef, name in zip(model.coef_, x_cols):
                    equation_str += f" + {coef:.4f} * {name}" if coef >= 0 else f" - {abs(coef):.4f} * {name}"
                st.write(f"**Equation:** `{y_col} = {equation_str}`")

                st.write("##### Residuals Plot")
                residuals = y_test - y_pred
                res_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                     title="Residuals vs. Predicted Values")
                res_fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(res_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during regression: {e}")

def perform_logistic_regression(data, numeric_columns, categorical_columns, test_size):
    st.write("---")
    st.write("#### Logistic Regression Report")
    
    if not categorical_columns:
        st.warning("Logistic Regression requires a categorical target variable.")
        return
    if not numeric_columns:
        st.warning("Logistic Regression requires at least one numeric feature column.")
        return

    binary_targets = [col for col in categorical_columns if data[col].nunique() == 2]
    if not binary_targets:
        st.warning("Logistic Regression is best suited for a binary target variable (with exactly 2 unique classes). No such column found.")
        return

    target_var = st.selectbox("Select Target Variable (Binary)", ["Select a column"] + binary_targets, key="log_reg_target")
    
    if target_var != "Select a column":
        st.write(f"Target Variable: **{target_var}**")
        feature_cols = [col for col in numeric_columns if col != target_var]
        
        # REMOVED: The .dropna() call.
        temp_data = data[feature_cols + [target_var]]

        if temp_data.shape[0] < 10:
            st.error("Not enough data for classification. A minimum of 10 rows is required.")
            return
            
        try:
            X = temp_data[feature_cols]
            y = temp_data[target_var]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            st.write("##### Model Performance on Test Set")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{acc:.2%}")
            m_col2.metric("Precision", f"{prec:.2%}")
            m_col3.metric("Recall", f"{rec:.2%}")
            m_col4.metric("F1-Score", f"{f1:.2%}")

            intercept = model.intercept_[0]
            coeffs = model.coef_[0]
            equation_str = f"{intercept:.4f}"
            for coef, name in zip(coeffs, X.columns):
                if coef >= 0:
                    equation_str += f" + {coef:.4f} * {name}"
                else:
                    equation_str += f" - {abs(coef):.4f} * {name}"

            st.write(f"**Model Equation (Log-Odds):** `z = {equation_str}`")
            st.write("**Model Equation (Sigmoid):** `P(Y=1) = 1 / (1 + exp(-z))`")

            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                    colorscale='Blues', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with v_col2:
                st.write("##### Feature Coefficients")
                coeffs = model.coef_[0]
                coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coeffs})
                coeff_df['Color'] = ['green' if c > 0 else 'red' for c in coeff_df['Coefficient']]
                coeff_df = coeff_df.sort_values(by="Coefficient", ascending=True)
                
                coeff_fig = px.bar(coeff_df, x="Coefficient", y="Feature", orientation='h', 
                                   title="Feature Influence on Outcome", color='Color',
                                   color_discrete_map={'green': '#2ca02c', 'red': '#d62728'})
                coeff_fig.update_layout(showlegend=False)
                st.plotly_chart(coeff_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_random_forest(data, numeric_columns, categorical_columns, test_size):
    st.write("---")
    st.write("#### Random Forest Classification Report")
    
    if not categorical_columns:
        st.warning("Random Forest Classification requires a categorical target variable.")
        return
    if not numeric_columns:
        st.warning("Random Forest requires at least one numeric feature column.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="rf_target")
    
    if target_var != "Select a column":
        st.write(f"Target Variable: **{target_var}**")
        feature_cols = [col for col in numeric_columns if col != target_var]
        
        # REMOVED: The .dropna() call.
        temp_data = data[feature_cols + [target_var]]

        if temp_data[target_var].nunique() < 2:
            st.error(f"The target variable '{target_var}' must have at least two unique classes for classification.")
            return
        if temp_data.shape[0] < 10:
            st.error("Not enough data for classification. A minimum of 10 rows is required.")
            return
            
        try:
            X = temp_data[feature_cols]
            y = temp_data[target_var]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
            
            n_estimators = st.slider("Number of Trees in Forest", 10, 500, 100, 10, key="rf_n_estimators")
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("##### Model Performance on Test Set")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{acc:.2%}")
            m_col2.metric("Precision", f"{prec:.2%}")
            m_col3.metric("Recall", f"{rec:.2%}")
            m_col4.metric("F1-Score", f"{f1:.2%}")

            st.write("**Core Concept:** `An ensemble of Decision Trees that votes on the final prediction.`")

            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                    colorscale='Blues', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with v_col2:
                st.write("##### Feature Importance")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=True)
                imp_fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h', text_auto='.2f', color="Importance", color_continuous_scale='Cividis')
                imp_fig.update_traces(textposition='inside', insidetextanchor='middle')
                imp_fig.update_layout(showlegend=False, margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(imp_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_decision_tree_classification(data, numeric_columns, categorical_columns, test_size):
    st.write("---")
    st.write("#### Decision Tree Classification Report")

    if not categorical_columns:
        st.warning("Decision Tree Classification requires a categorical target variable.")
        return
    if not numeric_columns:
        st.warning("Decision Tree requires at least one numeric feature column.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="dt_target")

    if target_var != "Select a column":
        st.write(f"Target Variable: **{target_var}**")
        feature_cols = [col for col in numeric_columns if col != target_var]
        
        # REMOVED: The .dropna() call.
        temp_data = data[feature_cols + [target_var]]

        if temp_data[target_var].nunique() < 2:
            st.error(f"The target variable '{target_var}' must have at least two unique classes for classification.")
            return
        if temp_data.shape[0] < 10:
            st.error("Not enough data for classification. A minimum of 10 rows is required.")
            return
            
        try:
            X = temp_data[feature_cols]
            y = temp_data[target_var]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_.astype(str)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
            
            max_depth = st.slider("Select Max Depth for the Tree", 2, 20, 5, key="dt_max_depth")
            
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("##### Model Performance on Test Set")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{acc:.2%}")
            m_col2.metric("Precision", f"{prec:.2%}")
            m_col3.metric("Recall", f"{rec:.2%}")
            m_col4.metric("F1-Score", f"{f1:.2%}")

            st.write("**Core Concept:** `A tree-like model of decisions based on feature values.`")

            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                    colorscale='Greens', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with v_col2:
                st.write("##### Feature Importance")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=True)
                imp_fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h', text_auto='.2f', color="Importance", color_continuous_scale='Greens')
                imp_fig.update_traces(textposition='inside', insidetextanchor='middle')
                imp_fig.update_layout(showlegend=False, margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(imp_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_svm_classification(data, numeric_columns, categorical_columns, test_size):
    st.write("---")
    st.write("#### Support Vector Machine (SVM) Report")

    if not categorical_columns:
        st.warning("SVM requires a categorical target variable.")
        return
    if not numeric_columns:
        st.warning("SVM requires at least one numeric feature column.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="svm_target")

    if target_var != "Select a column":
        feature_cols = [col for col in numeric_columns if col != target_var]
        
        # REMOVED: The .dropna() call.
        temp_data = data[feature_cols + [target_var]]

        if temp_data[target_var].nunique() < 2:
            st.error(f"The target variable '{target_var}' must have at least two unique classes for classification.")
            return
        if temp_data.shape[0] < 10:
            st.error("Not enough data for classification. A minimum of 10 rows is required.")
            return
            
        try:
            X = temp_data[feature_cols]
            y = temp_data[target_var]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            C = st.slider("Select Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
            
            model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            st.write("##### Model Performance on Test Set")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            if len(class_names) == 2:
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")
                m_col5.metric("AUC Score", f"{roc_auc:.3f}")
            else:
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Accuracy", f"{acc:.2%}")
                m_col2.metric("Precision", f"{prec:.2%}")
                m_col3.metric("Recall", f"{rec:.2%}")
                m_col4.metric("F1-Score", f"{f1:.2%}")

            st.write("**Model Equation (Hyperplane):** `w * x - b = 0`")
            st.write("**Kernel Trick Used:** `Radial Basis Function (RBF)`")
            
            if len(class_names) == 2:
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.write("##### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    cm_fig = go.Figure(data=go.Heatmap(
                        z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                        colorscale='Purples', reversescale=True,
                        hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                    ))
                    for i in range(len(class_names)):
                        for j in range(len(class_names)):
                            cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                    cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                    st.plotly_chart(cm_fig, use_container_width=True)

                with v_col2:
                    st.write("##### ROC Curve")
                    roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='purple', width=2), name=f'AUC = {roc_auc:.2f}'))
                    roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    roc_fig.update_layout(
                        title_text='Receiver Operating Characteristic (ROC) Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                    colorscale='Purples', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)
                st.info("ROC Curve is only available for binary classification tasks.")

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_knn_classification(data, numeric_columns, categorical_columns, test_size):
    st.write("---")
    st.write("#### K-Nearest Neighbors (KNN) Report")

    if not categorical_columns:
        st.warning("KNN requires a categorical target variable.")
        return
    if not numeric_columns:
        st.warning("KNN requires at least one numeric feature column.")
        return

    target_var = st.selectbox("Select Target Variable (Categorical)", ["Select a column"] + categorical_columns, key="knn_target")
    
    if target_var != "Select a column":
        feature_cols = st.multiselect("Select Feature Columns (at least 1)", [col for col in numeric_columns if col != target_var], default=[col for col in numeric_columns if col != target_var][:2], key="knn_features")

        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return
            
        # REMOVED: The .dropna() call.
        temp_data = data[feature_cols + [target_var]]

        if temp_data[target_var].nunique() < 2:
            st.error(f"The target variable '{target_var}' must have at least two unique classes for classification.")
            return
        if temp_data.shape[0] < 10:
            st.error("Not enough data for classification. A minimum of 10 rows is required.")
            return
            
        try:
            X = temp_data[feature_cols]
            y = temp_data[target_var]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_.astype(str)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            k_neighbors = st.slider("Select Number of Neighbors (k)", 1, 15, 5, key="knn_k")
            
            model = KNeighborsClassifier(n_neighbors=k_neighbors)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            st.write("##### Model Performance on Test Set")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{acc:.2%}")
            m_col2.metric("Precision", f"{prec:.2%}")
            m_col3.metric("Recall", f"{rec:.2%}")
            m_col4.metric("F1-Score", f"{f1:.2%}")

            st.write("**Core Concept (Euclidean Distance):** `d(p, q) = sqrt(sum((q_i - p_i)^2))`")

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.write("##### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"Actual: {c}" for c in class_names],
                    colorscale='Oranges', reversescale=True,
                    hovertemplate='Actual: %{y}<br>Predicted: {x}<br>Count: %{z}<extra></extra>'
                ))
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        cm_fig.add_annotation(x=f"Pred: {class_names[j]}", y=f"Actual: {class_names[i]}", text=str(cm[i, j]), showarrow=False, font={"color": "white" if cm[i,j] < cm.max()/1.5 else "black"})
                cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(cm_fig, use_container_width=True)

            with v_col2:
                st.write("##### Elbow Method for Optimal k")
                with st.spinner("Calculating optimal k..."):
                    error_rate = []
                    k_range = range(1, 16)
                    for i in k_range:
                        knn = KNeighborsClassifier(n_neighbors=i)
                        knn.fit(X_train_scaled, y_train)
                        pred_i = knn.predict(X_test_scaled)
                        error_rate.append(np.mean(pred_i != y_test))
                    
                    elbow_fig = go.Figure(data=go.Scatter(x=list(k_range), y=error_rate, mode='lines+markers', line=dict(color='#ff6347')))
                    elbow_fig.update_layout(title="Error Rate vs. K Value", xaxis_title="Number of Neighbors (k)", yaxis_title="Error Rate", margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(elbow_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

def perform_kmeans_clustering(data, numeric_columns):
    st.write("---")
    st.write("#### K-Means Clustering Report")
    if len(numeric_columns) < 2:
        st.warning("K-Means Clustering requires at least two numeric columns.")
        return

    kmeans_cols = st.multiselect("Select at least two columns for Clustering", numeric_columns, key="kmeans_cols")
    
    if len(kmeans_cols) >= 2:
        # REMOVED: The .dropna() call.
        clustering_data = data[kmeans_cols]

        if clustering_data.shape[0] < 10:
            st.error(f"Not enough data for clustering. At least 10 rows are required.")
            return

        st.write("**Objective Function (WCSS):** `minimize sum(||x - µ_i||^2)`")
        
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.write("##### Elbow Method for Optimal k")
            with st.spinner("Calculating optimal k..."):
                inertia = []
                k_range = range(1, 11)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans.fit(clustering_data)
                    inertia.append(kmeans.inertia_)
                
                elbow_fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertia, mode='lines+markers', line=dict(color='#ff6347')))
                elbow_fig.update_layout(title="Inertia vs. Number of Clusters", xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)", margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(elbow_fig, use_container_width=True)
            st.info("The 'elbow' in the plot above indicates a good balance for 'k'.")

        with v_col2:
            st.write("##### Cluster Visualization")
            num_clusters = st.number_input("Enter Number of Clusters (k) based on the plot", min_value=2, max_value=20, value=3, step=1, key="num_clusters")
            
            if clustering_data.shape[0] < num_clusters:
                st.error(f"Not enough data for the selected k. You need at least {num_clusters} rows, but only have {clustering_data.shape[0]}.")
                return
            
            try:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(clustering_data)
                centroids = kmeans.cluster_centers_
                
                plot_data = clustering_data.copy()
                plot_data["Cluster"] = labels
                
                cluster_fig = px.scatter(
                    plot_data, x=kmeans_cols[0], y=kmeans_cols[1], 
                    color=plot_data["Cluster"].astype(str), 
                    title=f"K-Means Clustering (k={num_clusters})",
                    labels={"color": "Cluster"},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                
                cluster_fig.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1],
                    mode='markers',
                    marker=dict(color='black', size=12, symbol='x', line=dict(width=2)),
                    name='Centroids'
                ))
                st.plotly_chart(cluster_fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
