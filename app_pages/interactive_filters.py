import streamlit as st
import pandas as pd

# Load the dataset
try:
    data = pd.read_csv('assets/AmesHousing.csv')
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'AmesHousing.csv' is in the 'assets' folder.")
    st.stop()

# Validate dataset structure
required_columns = ["Variable", "Units"]
required_variables = ["YearBuilt", "LotArea"]

if not all(col in data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    st.error(f"The dataset must contain the following columns: {', '.join(missing_cols)}")
    st.stop()

if not all(var in data["Variable"].values for var in required_variables):
    missing_vars = [var for var in required_variables if var not in data["Variable"].values]
    st.error(f"Missing required rows in the 'Variable' column: {', '.join(missing_vars)}")
    st.stop()

# Pivot the dataset
# Pivot the dataset
try:
    # Check for duplicates
    if data["Variable"].duplicated().any():
        st.warning("Duplicate variables found in the dataset. Removing duplicates...")
        data = data.drop_duplicates(subset=["Variable"], keep="first")
    
    # Pivot the data
    pivoted_data = data.pivot_table(
        index=None, 
        columns="Variable", 
        values="Units", 
        aggfunc="first"
    )

    # Clean up the pivoted DataFrame
    pivoted_data.columns.name = None
    pivoted_data.reset_index(drop=True, inplace=True)

        # Convert numeric ranges to integers for filtering
    for col in ["YearBuilt", "LotArea"]:
        pivoted_data[col] = pivoted_data[col].str.split(" - ").str[0].astype(int)
except Exception as e:
    st.error(f"Error pivoting dataset: {e}")
    st.stop()

# Convert numeric columns to appropriate data types
numeric_columns = [
    "YearBuilt",
    "LotArea"
]
for col in numeric_columns:
    pivoted_data[col] = pivoted_data[col].str.split(" - ").str[0].astype(int)

# Apply global filters
def apply_global_filters(data):
    # Filter: Year Built Range
    year_range = st.sidebar.slider(
        "Select Year Built Range", 
        min_value=data["YearBuilt"].min(), 
        max_value=data["YearBuilt"].max(), 
        value=(1950, 2000)
    )

    # Filter: Lot Area Range
    lot_area_range = st.sidebar.slider(
        "Select Lot Area Range (sq ft)", 
        min_value=data["LotArea"].min(), 
        max_value=data["LotArea"].max(), 
        value=(5000, 15000)
    )

    # Apply filters
    filtered_data = data[
        (data["YearBuilt"] >= year_range[0]) & 
        (data["YearBuilt"] <= year_range[1]) & 
        (data["LotArea"] >= lot_area_range[0]) & 
        (data["LotArea"] <= lot_area_range[1])
    ]

    st.sidebar.write(f"Filtered Properties: {len(filtered_data)}")
    return filtered_data

# Apply filters and display data
filtered_data = apply_global_filters(pivoted_data)
st.write(filtered_data)