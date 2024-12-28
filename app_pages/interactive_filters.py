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

# Pivot the data for easier filtering
try:
    # Ensure 'Units' column is numeric
    data["Units"] = pd.to_numeric(data["Units"], errors="coerce")
    
    # Pivot the dataset
    pivoted_data = data.pivot(index=None, columns="Variable", values="Units")
    
    # Drop rows with NaN in critical variables
    pivoted_data = pivoted_data.dropna(subset=required_variables)
except Exception as e:
    st.error(f"Error pivoting dataset: {e}")
    st.stop()

def apply_global_filters(pivoted_data):
    # Filter: Year Built Range
    year_range = st.sidebar.slider(
        "Select Year Built Range", 
        min_value=int(pivoted_data["YearBuilt"].min()), 
        max_value=int(pivoted_data["YearBuilt"].max()), 
        value=(1950, 2000)
    )

    # Filter: Lot Area Range
    lot_area_range = st.sidebar.slider(
        "Select Lot Area Range (sq ft)", 
        min_value=int(pivoted_data["LotArea"].min()), 
        max_value=int(pivoted_data["LotArea"].max()), 
        value=(5000, 15000)
    )

    # Apply filters to the data
    filtered_data = pivoted_data[
        (pivoted_data["YearBuilt"] >= year_range[0]) & 
        (pivoted_data["YearBuilt"] <= year_range[1]) & 
        (pivoted_data["LotArea"] >= lot_area_range[0]) & 
        (pivoted_data["LotArea"] <= lot_area_range[1])
    ]

    st.sidebar.write(f"Filtered Properties: {len(filtered_data)}")
    return filtered_data

# Apply the global filters
filtered_data = apply_global_filters(pivoted_data)

# Display filtered data
st.write("Filtered Data", filtered_data)
