import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_filters(data):
    st.header("Interactive Filters")

    # Add filters for price and number of bedrooms
    min_price, max_price = st.slider(
        "Select Price Range",
        min_value=int(data["SalePrice"].min()),
        max_value=int(data["SalePrice"].max()),
        value=(100000, 300000),
    )
    num_bedrooms = st.selectbox(
        "Select Number of Bedrooms",
        options=sorted(df["BedroomAbvGr"].unique())
    )

    # Filter the DataFrame
    filtered_data = data[
        (data["SalePrice"] >= min_price) &
        (data["SalePrice"] <= max_price) &
        (data["BedroomAbvGr"] == num_bedrooms)
    ]

    # Display filtered data
    st.write(f"Filtered results: {len(filtered_data)} properties found.")
    st.dataframe(filtered_data)
