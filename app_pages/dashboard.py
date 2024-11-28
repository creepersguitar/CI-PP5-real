import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def render_dashboard(df, model=None, X_test=None, y_test=None):
    """Render the entire dashboard using the cleaned dataframe and model evaluation."""
    # Visualize initial data
    st.subheader("Exploratory Data Analysis")
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Visualize missing values
    visualize_nan_distribution(df)

    # If model and test data are available, evaluate the model
    if model and X_test is not None and y_test is not None:
        evaluate_model(model, X_test, y_test)

def visualize_nan_distribution(df):
    """Plot NaN value distribution."""
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        st.write("### NaN Values Distribution")
        fig = px.bar(nan_counts, x=nan_counts.index, y=nan_counts.values, title='NaN Values per Column')
        st.plotly_chart(fig)
    else:
        st.write("No missing values found.")

def visualize_data(df):
    """Display exploratory data analysis visualizations."""
    st.subheader("Exploratory Data Analysis")

    # SalePrice Distribution
    if 'SalePrice' in df.columns:
        st.write("### Sale Price Distribution")
        fig = px.histogram(df, x='SalePrice', title='Sale Price Distribution')
        st.plotly_chart(fig)

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric data available for correlation heatmap.")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and display performance metrics and plots."""
    st.subheader("Model Performance Metrics")
    y_pred_rf = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred_rf)
    rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
    r2 = r2_score(y_test, y_pred_rf)

    # Display metrics
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.4f}")

    # Residuals Plot
    st.write("### Residuals Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test - y_pred_rf, alpha=0.5, edgecolor='b', s=50)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Residuals Plot', fontsize=16)
    ax.set_xlabel('Actual Sale Price', fontsize=14)
    ax.set_ylabel('Residuals', fontsize=14)
    st.pyplot(fig)
def model_performance_page(y_true, y_pred):
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.header("Model Performance")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("R-squared", f"{r2:.2f}")

    # Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    st.pyplot(plt)

    # Residual Plot
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    st.pyplot(plt)