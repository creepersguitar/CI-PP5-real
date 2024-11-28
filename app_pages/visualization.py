# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_prediction_vs_actual(y_test, y_pred):
    """
    Creates a scatter plot of Actual vs Predicted Prices.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # line of perfect prediction
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Prediction vs Actual Prices")
    return fig

def plot_residuals(y_test, y_pred):
    """
    Creates a residual plot.
    """
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', lw=2)  # Zero line for reference
    ax.set_xlabel("Predicted Prices")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    return fig

def display_model_performance_metrics(rmse, mae, r2):
    """
    Displays the model performance metrics in Streamlit.
    """
    st.title("Model Performance Metrics")

    st.subheader("Performance Metrics")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")
