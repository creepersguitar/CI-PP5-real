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

# Path to the CSV file inside the assets folder
file_path = 'assets/AmesHousing.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Ensure no leading or trailing spaces in column names
df.columns = df.columns.str.strip()

# Check the structure of the DataFrame
st.write("### Initial DataFrame Preview")
st.dataframe(df.head())  # Use Streamlit to display the DataFrame
st.write("Initial DataFrame columns:", df.columns)

# Check if the data is in a pivoted format (metadata-style)
if 'Variable' in df.columns:
    # Pivot the DataFrame to get features as columns
    if 'Units' in df.columns:
        df_pivot = df.pivot(index=None, columns='Variable', values='Units')
        # Reset index if needed
        df_pivot.reset_index(drop=True, inplace=True)
        df = df_pivot  # Update df to the pivoted DataFrame
        st.write("Data reshaped successfully.")
    else:
        st.write("The 'Units' column is missing; unable to reshape data.")
else:
    st.write("Data is not in a metadata format. Proceeding with the original DataFrame.")

# Check the DataFrame shape and columns after reshaping
st.write("### DataFrame Shape and Columns After Reshaping")
st.write(f"Shape: {df.shape}")
st.write("Columns:", df.columns)

# Verify if the required columns are present for TotalSF calculation
required_columns = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'SalePrice']
missing_columns = [col for col in required_columns if col not in df.columns]

if not missing_columns:
    # Calculate TotalSF
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    st.write("TotalSF column successfully created.")
else:
    st.write(f"The following required columns are missing for TotalSF calculation: {missing_columns}")

# Convert categorical features to numeric if needed
if 'OverallQual' in df.columns and df['OverallQual'].dtype == 'object':
    st.write("Unique values in OverallQual before encoding:", df['OverallQual'].unique())
    le = LabelEncoder()
    df['OverallQual'] = le.fit_transform(df['OverallQual'])

# Clean the YearBuilt column if necessary
if 'YearBuilt' in df.columns:
    st.write("Unique values in YearBuilt before cleaning:", df['YearBuilt'].unique())
    df['YearBuilt'] = pd.to_numeric(df['YearBuilt'].str.split(' - ').str[0].str.replace(',', ''), errors='coerce')
    st.write("Unique values in YearBuilt after cleaning:", df['YearBuilt'].unique())

# Check for NaN values in the necessary columns
st.write("### Check for NaN Values")
nan_counts = df[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt', 'SalePrice']].isnull().sum()
st.write("NaN Counts in Important Columns:")
st.write(nan_counts)

# Fill or drop NaN values in critical columns
# Option 1: Drop rows with NaN values
df.dropna(subset=['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt', 'SalePrice'], inplace=True)

# Option 2: Alternatively, you could fill missing values instead
# df.fillna(df.median(), inplace=True)

# Check if the columns still exist
if 'TotalSF' in df.columns and 'SalePrice' in df.columns:
    X = df[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt']]
    y = df['SalePrice']
else:
    st.write("TotalSF or SalePrice column is not available. Skipping related operations.")
    # Use a subset of features that are available
    available_columns = [col for col in ['OverallQual', 'GarageArea', 'YearBuilt'] if col in df.columns]
    if 'SalePrice' in df.columns and available_columns:
        X = df[available_columns]
        y = df['SalePrice']
    else:
        st.write("Insufficient data for model training. Exiting.")
        X, y = None, None

# Proceed if X and y are valid
if X is not None and y is not None:
    # Handle NaN values
    X = X.dropna()
    y = y[X.index]  # Align y with X after dropping

    # Split into training and testing datasets
    if X.shape[0] > 0 and y.shape[0] > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Title of the dashboard
        st.title("Heritage Housing Price Prediction Dashboard")
        st.write("This dashboard helps predict house prices in Ames, Iowa using Exploratory Data Analysis and machine learning models.")
        st.write("### Ames Housing Dataset Preview:")
        st.dataframe(df.head())
        st.write(f"Number of Rows: {df.shape[0]}, Number of Columns: {df.shape[1]}")
        st.subheader('Exploratory Data Analysis')

        # Display DataFrame shape and missing values
        st.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        missing_values = df.isnull().sum()
        st.write("Missing Values in Each Column:")
        st.write(missing_values)

        # Correlation Heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

        # Scatter plots and analysis if columns exist
        if 'TotalSF' in df.columns and 'SalePrice' in df.columns:
            st.write("### Total Square Footage vs Sale Price")
            fig = px.scatter(df, x='TotalSF', y='SalePrice', opacity=0.5, title='Total Square Footage vs Sale Price')
            st.plotly_chart(fig)

        if 'OverallQual' in df.columns:
            st.write("### Overall Quality vs Sale Price")
            fig = px.scatter(df, x='OverallQual', y='SalePrice', opacity=0.5, title='Overall Quality vs Sale Price')
            st.plotly_chart(fig)

        if 'YearBuilt' in df.columns:
            st.write("### Year Built vs Sale Price")
            fig = px.scatter(df, x='YearBuilt', y='SalePrice', opacity=0.5, title='Year Built vs Sale Price')
            st.plotly_chart(fig)

        # Model Evaluation
        st.subheader("Model Evaluation")
        y_pred_rf = rf_model.predict(X_test)
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
else:
    st.write("No sufficient data for model training.")
