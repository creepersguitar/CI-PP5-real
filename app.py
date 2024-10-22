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
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    logging.info(f"Loaded data with columns: {df.columns.tolist()}")
    return df

# Function to clean the data
def clean_data(df):
    # Log initial DataFrame shape and columns
    logging.info(f"Initial DataFrame shape: {df.shape}")
    logging.info(f"Initial columns: {df.columns.tolist()}")

    # Check if 'Units' column exists
    if 'Units' not in df.columns:
        logging.error("Column 'Units' is missing from the DataFrame.")
        raise KeyError("Column 'Units' is missing from the DataFrame.")

    # Handle non-numeric entries in 'Units' column
    def convert_to_numeric(value):
        try:
            # If the value is a range like "0 - 1418", split and take the average
            if '-' in str(value):
                start, end = map(float, value.split('-'))
                return (start + end) / 2
            # Otherwise, try converting directly to float
            return float(value)
        except ValueError:
            return np.nan

    # Apply the conversion function to the 'Units' column
    df['Units'] = df['Units'].apply(convert_to_numeric)

    # Check for required columns in the DataFrame
    required_vars = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
    for col in required_vars:
        if col not in df['Variable'].values:
            logging.warning(f"Column {col} is missing from the DataFrame.")
    
    # Ensure the required numeric columns are present and convert to numeric
    for col in required_vars:
        if col in df['Variable'].values:
            df.loc[df['Variable'] == col, 'Units'] = pd.to_numeric(df.loc[df['Variable'] == col, 'Units'], errors='coerce')

    # Calculate TotalSF if required variables are present
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

    # Log the DataFrame after cleaning
    logging.info(f"DataFrame shape after cleaning: {df.shape}")
    logging.info(f"Columns after cleaning: {df['Variable'].unique()}")

    return df

# Function to visualize data
def visualize_data(df):
    st.subheader("Exploratory Data Analysis")

    # Distribution of SalePrice
    if 'SalePrice' in df.columns:
        fig = px.histogram(df, x='SalePrice', title='Sale Price Distribution')
        st.plotly_chart(fig)

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# Function to train the model
def train_model(X, y):
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

    # Reshape the DataFrame so 'Variable' values become columns
    df_pivot = df.pivot(index=None, columns='Variable', values='Units').reset_index(drop=True)
    
    # Log the reshaped DataFrame
    logging.info(f"DataFrame after reshaping: {df_pivot.shape}")
    logging.info(f"Columns after reshaping: {df_pivot.columns.tolist()}")

    # Check for critical columns after cleaning
    required_columns = ['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt', 'SalePrice']
    missing_columns = [col for col in required_columns if col not in df_pivot.columns]
    
    if missing_columns:
        st.write(f"Missing critical columns: {missing_columns}. Please check the data.")
        logging.warning(f"Missing critical columns: {missing_columns}")
        return  # Exit if critical columns are missing

    # Check for NaN values in critical columns
    st.write("### Check for NaN Values After Cleaning")
    nan_counts = df_pivot[required_columns].isnull().sum()
    st.write("NaN Counts in Important Columns:")
    st.write(nan_counts)

    # Fill NaN values
    df_pivot.fillna({
        'TotalSF': df_pivot['TotalSF'].median(),
        'OverallQual': df_pivot['OverallQual'].median(),  # Use median for numerical fill
        'GarageArea': df_pivot['GarageArea'].median(),
        'YearBuilt': df_pivot['YearBuilt'].median(),
        'SalePrice': df_pivot['SalePrice'].median()
    }, inplace=True)

    # Re-check for NaN values after filling
    nan_counts_after = df_pivot[required_columns].isnull().sum()
    st.write("NaN Counts in Important Columns After Filling:")
    st.write(nan_counts_after)

    # Proceed if critical columns are filled
    if nan_counts_after.sum() == 0:
        X = df_pivot[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt']]
        y = df_pivot['SalePrice']
    else:
        st.write("Still have NaN values in critical columns. Exiting.")
        return  # Exit if any critical columns still have NaNs

    # Proceed if X and y are valid
    model, X_test, y_test = train_model(X, y)

    # Visualize the data
    visualize_data(df_pivot)

    # Model Evaluation
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
