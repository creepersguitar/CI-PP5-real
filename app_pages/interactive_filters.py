import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_filters(data):
    st.header("Interactive Filters")

    # Simulate a dataset for testing
    dummy_data = {
        "Property ID": range(1, 11),
        "OverallQual": [7, 8, 6, 5, 9, 8, 4, 3, 10, 2],
        "SalePrice": [450000, 500000, 350000, 300000, 600000, 480000, 280000, 200000, 750000, 150000],
        "GrLivArea": [2000, 2500, 1800, 1600, 3000, 2400, 1400, 1200, 3500, 1000],
    }
    market_df = pd.DataFrame(dummy_data)

    # Interactive filters
    st.sidebar.header("Filters")

    # Filter by SalePrice
    min_price, max_price = st.sidebar.slider(
        "Sale Price Range ($):",
        min_value=int(market_df["SalePrice"].min()),
        max_value=int(market_df["SalePrice"].max()),
        value=(int(market_df["SalePrice"].min()), int(market_df["SalePrice"].max()))
    )

    # Filter by OverallQual
    min_qual, max_qual = st.sidebar.slider(
        "Overall Quality Range:",
        min_value=int(market_df["OverallQual"].min()),
        max_value=int(market_df["OverallQual"].max()),
        value=(int(market_df["OverallQual"].min()), int(market_df["OverallQual"].max()))
    )

    # Apply filters
    filtered_data = market_df[
        (market_df["SalePrice"] >= min_price) &
        (market_df["SalePrice"] <= max_price) &
        (market_df["OverallQual"] >= min_qual) &
        (market_df["OverallQual"] <= max_qual)
    ]

    # Display filtered data
    st.subheader("Filtered Properties")
    st.write(filtered_data)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(filtered_data.describe())

    # Visualization
    st.subheader("Price Distribution")
    st.bar_chart(filtered_data["SalePrice"])