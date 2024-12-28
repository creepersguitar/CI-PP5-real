import streamlit as st
import matplotlib.pyplot as plt
from app_pages.interactive_filters import apply_global_filters

fdata = None

def initialize(input_data):
    global fdata
    fdata = apply_global_filters(input_data)

# Function to display model performance
def show_metrics():
    st.header("Model Performance Metrics")

    # Display model performance metrics (Replace with actual results)
    rmse = 23456.78
    mae = 15678.90
    r2 = 0.89

    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse}")
    st.write(f"**Mean Absolute Error (MAE)**: {mae}")
    st.write(f"**R-squared (R²)**: {r2}")

    # Visualize Performance - Actual vs Predicted plot
    st.subheader("Actual vs Predicted Plot")
    # Replace this with actual data
    actual_values = [200000, 150000, 250000]
    predicted_values = [195000, 148000, 245000]
    
    plt.scatter(actual_values, predicted_values)
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.title('Actual vs Predicted Sale Price')
    st.pyplot(plt)

    # Add residual plot
    st.subheader("Residual Plot")
    residuals = [actual - pred for actual, pred in zip(actual_values, predicted_values)]
    plt.scatter(actual_values, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual Sale Price')
    st.pyplot(plt)
