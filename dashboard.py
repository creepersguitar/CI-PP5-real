import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Example predicted vs. actual data
actual = [150000, 200000, 300000]
predicted = [145000, 210000, 310000]

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

# Display metrics in Streamlit
st.subheader("Model Performance Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Function to visualize data
def visualize_data(df):
    """Display exploratory data analysis visualizations."""
    st.subheader("Exploratory Data Analysis")

    # Distribution of NaN values
    st.write("### NaN Values Distribution")
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        fig = px.bar(nan_counts, x=nan_counts.index, y=nan_counts.values, title='NaN Values per Column')
        st.plotly_chart(fig)
    else:
        st.write("No missing values found.")

    # Display the distribution of SalePrice
    if 'SalePrice' in df.columns:
        fig = px.histogram(df, x='SalePrice', title='Sale Price Distribution')
        st.plotly_chart(fig)

    # Display a correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric data available for correlation heatmap.")

def evaluate_model(model, X_test, y_test):
    st.subheader("Model Evaluation")
    y_pred_rf = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_rf)
    rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
    st.write(f"### R²: {r2:.4f}")
    st.write(f"### RMSE: {rmse:.2f}")
    # Plot residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test - y_pred_rf, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    st.pyplot(fig)