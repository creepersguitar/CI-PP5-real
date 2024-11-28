import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from app_pages.dashboard import render_dashboard, model_performance_page
# Set up loggin
logging.basicConfig(level=logging.INFO)

# Function to load the dataset
def load_data(file_path):
    """Load the dataset and clean up column names."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    logging.info(f"Loaded data with columns: {df.columns.tolist()}")
    return df

# Function to clean the data
def clean_data(df):
    """Clean the data by converting non-numeric values and checking for required variables."""
    logging.info(f"Initial DataFrame shape: {df.shape}")
    logging.info(f"Initial columns: {df.columns.tolist()}")

    # Check if 'Units' column exists
    if 'Units' not in df.columns:
        logging.error("Column 'Units' is missing from the DataFrame.")
        raise KeyError("Column 'Units' is missing from the DataFrame.")

    # Handle non-numeric entries in 'Units' column
    def convert_to_numeric(value):
        """Convert range values to average numeric values, and other values to float."""
        try:
            if '-' in str(value):  # Handles range-like values "0 - 1418"
                start, end = map(float, value.split('-'))
                return (start + end) / 2
            return float(value)
        except ValueError:
            return np.nan

    # Apply conversion to the 'Units' column
    df['Units'] = df['Units'].apply(convert_to_numeric)

    # Required variables for TotalSF calculation
    required_vars = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
    for col in required_vars:
        if col not in df['Variable'].values:
            logging.warning(f"Column {col} is missing from the DataFrame.")

    # Convert the 'Units' column to numeric for the required variables
    for col in required_vars:
        if col in df['Variable'].values:
            df.loc[df['Variable'] == col, 'Units'] = pd.to_numeric(df.loc[df['Variable'] == col, 'Units'], errors='coerce')

    # Calculate TotalSF if all required variables are present
    if all(var in df['Variable'].values for var in required_vars):
        first_flr_sf = df.loc[df['Variable'] == '1stFlrSF', 'Units'].fillna(0).values[0]
        second_flr_sf = df.loc[df['Variable'] == '2ndFlrSF', 'Units'].fillna(0).values[0]
        total_bsmt_sf = df.loc[df['Variable'] == 'TotalBsmtSF', 'Units'].fillna(0).values[0]

        total_sf = first_flr_sf + second_flr_sf + total_bsmt_sf
        
        # Append TotalSF to DataFrame
        df = df.append({'Variable': 'TotalSF', 'Units': total_sf}, ignore_index=True)
        logging.info("TotalSF column created successfully.")
    else:
        logging.warning("Required variables for TotalSF calculation are missing.")

    logging.info(f"DataFrame shape after cleaning: {df.shape}")
    logging.info(f"Columns after cleaning: {df['Variable'].unique()}")
    return df

# Function to train the model
def train_model(X, y):
    """Train the Random Forest model and return the model and test data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test
def mfp(y_true, y_pred):
    # Example menu routing
    menu = ["Home", "Model Performance"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Welcome to the House Price Predictor")
    elif choice == "Model Performance":
        # Pass your actual and predicted values here
        y_true = [100, 200, 300]  # Replace with actual values
        y_pred = [110, 190, 295]  # Replace with predicted values
        model_performance_page(y_true, y_pred)
def main():
    st.title("Heritage Housing Price Prediction Dashboard")
    # Load and clean data
    file_path = 'assets/AmesHousing.csv'
    df = load_data(file_path)
    df = clean_data(df)
    
    # Pass the dataframe to render_dashboard
    render_dashboard(df)

if __name__ == "__main__":
    main()