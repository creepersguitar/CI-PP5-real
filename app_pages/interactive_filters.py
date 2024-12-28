import streamlit as st
import pandas as pd

# Load the dataset
try:
    data = pd.read_csv('assets/AmesHousing.csv')
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'AmesHousing.csv' is in the 'assets' folder.")
    st.stop()

# Ensure required columns exist in the dataset
required_columns = ["YearBuilt", "LotArea"]
if not all(col in data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    st.error(f"Missing required columns in the dataset: {', '.join(missing_cols)}")
    st.stop()

# Ensure data types are numeric for filtering
try:
    data["YearBuilt"] = pd.to_numeric(data["YearBuilt"], errors="coerce")
    data["LotArea"] = pd.to_numeric(data["LotArea"], errors="coerce")
except Exception as e:
    st.error(f"Error converting columns to numeric: {e}")
    st.stop()

def apply_global_filters(data):
    # Drop rows with NaN values in critical columns
    data = data.dropna(subset=["YearBuilt", "LotArea"])

    # Filter: Year Built Range
    year_range = st.sidebar.slider(
        "Select Year Built Range", 
        min_value=int(data["YearBuilt"].min()), 
        max_value=int(data["YearBuilt"].max()), 
        value=(1950, 2000)
    )

    # Filter: Lot Area Range
    lot_area_range = st.sidebar.slider(
        "Select Lot Area Range (sq ft)", 
        min_value=int(data["LotArea"].min()), 
        max_value=int(data["LotArea"].max()), 
        value=(5000, 15000)
    )

    # Apply filters to the data
    filtered_data = data[
        (data["YearBuilt"] >= year_range[0]) & 
        (data["YearBuilt"] <= year_range[1]) & 
        (data["LotArea"] >= lot_area_range[0]) & 
        (data["LotArea"] <= lot_area_range[1])
    ]

    st.sidebar.write(f"Filtered Properties: {len(filtered_data)}")
    return filtered_data

# Apply the global filters
filtered_data = apply_global_filters(data)

# Display filtered data for debugging purposes
st.write("Filtered Data", filtered_data)
