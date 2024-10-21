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
from sklearn.preprocessing import LabelEncoder
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

    # Convert numeric columns if they exist
    numeric_columns = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'SalePrice', 'YearBuilt']
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
            except Exception as e:
                logging.error(f"Error converting {col}: {e}")
        else:
            logging.warning(f"Column {col} is missing from the DataFrame.")

    # Calculate TotalSF if required columns are present
    required_columns = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
    if all(col in df.columns for col in required_columns):
        df['TotalSF'] = df['1stFlrSF'].fillna(0) + df['2ndFlrSF'].fillna(0) + df['TotalBsmtSF'].fillna(0)
        logging.info("TotalSF column created successfully.")
    else:
        logging.warning("Required columns for TotalSF calculation are missing.")
    
    # Convert OverallQual to numeric if it exists
    if 'OverallQual' in df.columns:
        if df['OverallQual'].dtype == 'object':
            le = LabelEncoder()
            df['OverallQual'] = le.fit_transform(df['OverallQual'])
            logging.info("OverallQual column encoded.")
        else:
            logging.info("OverallQual column is already numeric.")
    
    # Clean YearBuilt column
    if 'YearBuilt' in df.columns:
        df['YearBuilt'] = pd.to_numeric(df['YearBuilt'].astype(str).str.split(' - ').str[0].str.replace(',', ''), errors='coerce')
        logging.info("YearBuilt column cleaned.")
    else:
        logging.warning("YearBuilt column is missing from the DataFrame.")
    
    # Log final columns after cleaning
    logging.info(f"Columns after cleaning: {df.columns.tolist()}")

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

    # Check for critical rows after cleaning
    required_rows = ['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt', 'SalePrice']
    missing_rows = [row for row in required_rows if row not in df['Variable'].values]
    
    if missing_rows:
        st.write(f"Missing critical rows: {missing_rows}. Please check the data.")
        logging.warning(f"Missing critical rows: {missing_rows}")
        return  # Exit if critical rows are missing

    # Log the DataFrame shape and columns
    logging.info(f"DataFrame shape after cleaning: {df.shape}")
    logging.info(f"Columns available: {df.columns.tolist()}")

    # Check for NaN values in critical columns
    st.write("### Check for NaN Values After Cleaning")
    nan_counts = df[required_rows].isnull().sum()
    st.write("NaN Counts in Important Rows:")
    st.write(nan_counts)

    # Fill NaN values
    df.fillna({
        'TotalSF': df['TotalSF'].median(),
        'OverallQual': df['OverallQual'].mode()[0],
        'GarageArea': df['GarageArea'].median(),
        'YearBuilt': df['YearBuilt'].median(),
        'SalePrice': df['SalePrice'].median()
    }, inplace=True)

    # Re-check for NaN values after filling
    nan_counts_after = df[required_rows].isnull().sum()
    st.write("NaN Counts in Important Rows After Filling:")
    st.write(nan_counts_after)

    # Proceed if critical rows are filled
    if df[['GarageArea', 'YearBuilt', 'SalePrice']].isnull().sum().sum() == 0:
        X = df[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt']]
        y = df['SalePrice']
    else:
        st.write("Still have NaN values in critical rows. Exiting.")
        X, y = None, None

    # Proceed if X and y are valid
    if X is not None and y is not None:
        model, X_test, y_test = train_model(X, y)

        # Visualize the data
        visualize_data(df)

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
    else:
        st.write("No sufficient data for model training.")

# Run the app
if __name__ == "__main__":
    main()
