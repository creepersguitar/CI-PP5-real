import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Path to the CSV file inside the assets folder
file_path = '../assets/AmesHousing.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Ensure no leading or trailing spaces in column names
df.columns = df.columns.str.strip()

# Check the structure of the DataFrame
print("Initial DataFrame preview:\n", df.head())
print("Initial DataFrame columns:", df.columns)

# Reshape the DataFrame: Pivot the variable names into columns
# Assuming the DataFrame has 'Variable' and 'Value' columns
df_pivot = df.pivot(index=None, columns='Variable', values='Units')

# Reset index if needed
df_pivot.reset_index(drop=True, inplace=True)

# Display the reshaped DataFrame for verification
print("Reshaped DataFrame preview:\n", df_pivot.head())
print("Columns after pivoting:", df_pivot.columns)

# Fill missing values with the median
df_pivot.fillna(df_pivot.median(), inplace=True)

# Convert categorical features to numeric
# If 'OverallQual' or other categorical columns have not been converted, do so
if 'OverallQual' in df_pivot.columns and df_pivot['OverallQual'].dtype == 'object':
    print("Unique values in OverallQual before encoding:", df_pivot['OverallQual'].unique())
    le = LabelEncoder()
    df_pivot['OverallQual'] = le.fit_transform(df_pivot['OverallQual'])

# Check the YearBuilt column for issues
if 'YearBuilt' in df_pivot.columns:
    print("Unique values in YearBuilt before cleaning:", df_pivot['YearBuilt'].unique())
    # Clean the YearBuilt column
    df_pivot['YearBuilt'] = pd.to_numeric(df_pivot['YearBuilt'].str.split(' - ').str[0], errors='coerce')
    print("Unique values in YearBuilt after cleaning:", df_pivot['YearBuilt'].unique())

# Add new feature for total square footage if the relevant columns exist
required_columns = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
if all(col in df_pivot.columns for col in required_columns):
    df_pivot['TotalSF'] = df_pivot['1stFlrSF'] + df_pivot['2ndFlrSF'] + df_pivot['TotalBsmtSF']
else:
    print("One or more required columns are missing for TotalSF calculation.")

# Specify features and target variable
X = df_pivot[['TotalSF', 'OverallQual', 'GarageArea', 'YearBuilt']]
y = df_pivot['SalePrice']

# Check the shapes of X and y before proceeding
print("Shape of X before NaN handling:", X.shape)
print("Shape of y before NaN handling:", y.shape)

# Check for NaN values in features and target variable
print("NaN values in X:", X.isnull().sum())
print("NaN values in y:", y.isnull().sum())

# Handle NaN values: Option 1: Drop rows with NaN values
X = X.dropna()
y = y[X.index]  # Align y with X after dropping

# Alternatively, you can fill NaN values instead of dropping
# X.fillna(X.median(), inplace=True)

# Check the shape after dropping NaNs
print("Shape of X after NaN handling:", X.shape)
print("Shape of y after NaN handling:", y.shape)

# Check if there are any samples left to split
if X.shape[0] == 0 or y.shape[0] == 0:
    print("No samples left for training and testing. Please check your data handling steps.")
else:
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Title of the dashboard
    st.title("Heritage Housing Price Prediction Dashboard")

    # Write introduction text
    st.write("This dashboard helps predict house prices in Ames, Iowa using Exploratory Data Analysis and machine learning models.")

    # Display the first few rows of the dataset for reference
    st.write("### Ames Housing Dataset Preview:")
    st.dataframe(df_pivot.head())

    # Display the shape of the dataset (rows and columns)
    st.write(f"Number of Rows: {df_pivot.shape[0]}, Number of Columns: {df_pivot.shape[1]}")
st.subheader('Exploratory Data Analysis')

# Display the shape of the DataFrame
st.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Check for missing values
missing_values = df.isnull().sum()
st.write("Missing Values in Each Column:")
st.write(missing_values)

# Check if the DataFrame is empty
if df.empty:
    st.write("The DataFrame is empty. Please check the data loading process.")
else:
    # Try to select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Check if there are any numeric columns to calculate the correlation
    if numeric_df.empty:
        st.write("No numeric columns available to calculate correlation.")
    else:
        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Ensure 'TotalSF' and 'SalePrice' exist in DataFrame
        if 'TotalSF' in df.columns and 'SalePrice' in df.columns:
            # Scatter plot for Total Square Footage vs Sale Price
            st.write("### Total Square Footage vs Sale Price")
            fig = px.scatter(df, x='TotalSF', y='SalePrice', opacity=0.5, title='Total Square Footage vs Sale Price')
            st.plotly_chart(fig)

            # Box plot for Neighborhood vs Sale Price
            if 'Neighborhood' in df.columns:
                st.write("### Neighborhood vs Sale Price")
                fig = px.box(df, x='Neighborhood', y='SalePrice', title='Neighborhood vs Sale Price')
                st.plotly_chart(fig)
            else:
                st.write("Column 'Neighborhood' not found in the dataset.")

            # Scatter plot for Overall Quality vs Sale Price
            if 'OverallQual' in df.columns:
                st.write("### Overall Quality vs Sale Price")
                fig = px.scatter(df, x='OverallQual', y='SalePrice', opacity=0.5, title='Overall Quality vs Sale Price')
                st.plotly_chart(fig)
            else:
                st.write("Column 'OverallQual' not found in the dataset.")

            # Scatter plot for Year Built vs Sale Price
            if 'YearBuilt' in df.columns:
                st.write("### Year Built vs Sale Price")
                fig = px.scatter(df, x='YearBuilt', y='SalePrice', opacity=0.5, title='Year Built vs Sale Price')
                st.plotly_chart(fig)
            else:
                st.write("Column 'YearBuilt' not found in the dataset.")

            # Scatter plot for Garage Area vs Sale Price
            if 'GarageArea' in df.columns:
                st.write("### Garage Area vs Sale Price")
                fig = px.scatter(df, x='GarageArea', y='SalePrice', opacity=0.5, title='Garage Area vs Sale Price')
                st.plotly_chart(fig)
            else:
                st.write("Column 'GarageArea' not found in the dataset.")
# Check if 'TotalSF' exists, and if not, notify the user
if 'TotalSF' not in df.columns:
    st.error("The column 'TotalSF' is not found in the dataset. Please check the column names.")
else:
    # Sidebar for user input
    st.sidebar.header("Input Features")
    total_sf = st.sidebar.slider('Total Square Footage', int(df['TotalSF'].min()), int(df['TotalSF'].max()), int(df['TotalSF'].mean()))
    overall_qual = st.sidebar.slider('Overall Quality', int(df['OverallQual'].min()), int(df['OverallQual'].max()), int(df['OverallQual'].mean()))
    garage_area = st.sidebar.slider('Garage Area', int(df['GarageArea'].min()), int(df['GarageArea'].max()), int(df['GarageArea'].mean()))
    year_built = st.sidebar.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), int(df['YearBuilt'].mean()))

    # Predict house price based on input
    if st.sidebar.button("Predict Sale Price"):
        try:
            price = predict_price(total_sf, overall_qual, garage_area, year_built)
            st.write(f"### Predicted Sale Price: ${price:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
# Model Evaluation Section
st.subheader("Model Evaluation")
st.write("### Residuals Plot")

# Generate predictions
y_pred_rf = rf_model.predict(X_test)

# Create the residuals plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test - y_pred_rf, alpha=0.5, edgecolor='b', s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residuals Plot', fontsize=16)
ax.set_xlabel('Actual Sale Price', fontsize=14)
ax.set_ylabel('Residuals', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Display the plot in Streamlit
st.pyplot(fig)

# Display R² and RMSE for the model
r2 = r2_score(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

st.write(f"### Random Forest R²: {r2:.4f}")
st.write(f"### Random Forest RMSE: {rmse:.2f}")

# Provide additional interpretation
st.write("""
**Interpretation of the Residuals Plot**:
- The residuals should ideally be randomly scattered around zero without any clear pattern.
- If the residuals show a pattern (e.g., a curve), it might indicate that the model is not capturing some information in the data.
""")
