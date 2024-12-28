import streamlit as st
import pandas as pd

# Load the dataset
try:
    data = pd.read_csv('assets/AmesHousing.csv')
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'AmesHousing.csv' is in the 'assets' folder.")
    st.stop()

# Validate dataset structure
required_variables = ["YearBuilt", "LotArea"]
if "Variable" not in data.columns or "Units" not in data.columns:
    st.error("The dataset must contain 'Variable' and 'Units' columns.")
    st.stop()

if not all(var in data["Variable"].values for var in required_variables):
    missing_vars = [var for var in required_variables if var not in data["Variable"].values]
    st.error(f"Missing required rows in the 'Variable' column: {', '.join(missing_vars)}")
    st.stop()

# Pivot the data for easier filtering (Variable as columns, Value as rows)
try:
    pivoted_data = data.pivot(index=None, columns="Variable", values="Units")
    pivoted_data = pivoted_data.apply(pd.to_numeric, errors="coerce")  # Convert values to numeric
except Exception as e:
    st.error(f"Error pivoting dataset: {e}")
    st.stop()

def apply_global_filters(pivoted_data):
    # Drop rows with NaN values in critical columns
    pivoted_data = pivoted_data.dropna(subset=required_variables)

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

# Display filtered data for debugging purposes
st.write("Filtered Data", filtered_data)
