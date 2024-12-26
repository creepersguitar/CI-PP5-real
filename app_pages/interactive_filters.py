import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def apply_global_filters(data):
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