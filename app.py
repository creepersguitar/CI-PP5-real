import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
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

# Function to train the model
def train_model(X, y):
    """Train the Random Forest model and return the model and test data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test

# Main app execution
def main():
    st.title("Heritage Housing Price Prediction Dashboard")
    
    # Path to the CSV file inside the assets folder
    file_path = 'assets/AmesHousing.csv'

    # Load the dataset
    df = load_data(file_path)
    st.write("### Initial DataFrame Preview")
    st.dataframe(df.head())

    # Clean the data
    df = clean_data(df)

    # Reshape DataFrame so 'Variable' values become columns
    df_pivot = df.pivot(index=None, columns='Variable', values='Units').reset_index(drop=True)
    logging.info(f"DataFrame shape after reshaping: {df_pivot.shape}")
    logging.info(f"Columns after reshaping: {df_pivot.columns.tolist()}")

    # Check for critical columns after reshaping
    required_columns = ['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt', 'SalePrice']
    missing_columns = [col for col in required_columns if col not in df_pivot.columns]
    
    if missing_columns:
        st.write(f"Missing critical columns: {missing_columns}. Please check the data.")
        logging.warning(f"Missing critical columns: {missing_columns}")
        return  # Exit if critical columns are missing

    # Visualize NaN values before filling
    visualize_data(df_pivot)

    # Fill NaN values with median for numerical columns
    df_pivot.fillna(df_pivot.median(), inplace=True)

    # Check for remaining NaN values after filling
    nan_counts_after = df_pivot.isna().sum()
    nan_counts_after = nan_counts_after[nan_counts_after > 0]
    if not nan_counts_after.empty:
        st.write("### Columns with Remaining NaN Values After Filling")
        st.write(nan_counts_after)
        logging.error("Some columns still contain NaN values after filling.")
        return  # Exit if any columns still have NaNs

    # Prepare features and target variable for model training
    X = df_pivot[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt']]
    y = df_pivot['SalePrice']

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Model evaluation metrics
    st.subheader("Model Evaluation")
    y_pred_rf = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test - y_pred_rf, alpha=0.5, edgecolor='b', s=50)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Residuals Plot', fontsize=16)
    ax.set_xlabel('Actual Sale Price', fontsize=14)
    ax.set_ylabel('Residuals', fontsize=14)
    st.pyplot(fig)

    r2 = r2_score(y_test, y_pred_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    st.write(f"### Random Forest R²: {r2:.4f}")
    st.write(f"### Random Forest RMSE: {rmse:.2f}")

# Run the app
if __name__ == "__main__":
    main()
