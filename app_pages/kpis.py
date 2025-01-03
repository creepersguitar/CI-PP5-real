import streamlit as st
import pandas as pd
from app_pages.interactive_filters import apply_global_filters

fdata = None

def initialize(input_data):
    global fdata
    fdata = apply_global_filters(input_data)

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_kpis(data):
    st.header("Key Performance Indicators")
    
    # Extract the row corresponding to 'SalePrice'
    sale_price_row = data[data["Variable"] == "SalePrice"]
    
    if not sale_price_row.empty:
        # Parse the range in the 'Units' column to extract min and max prices
        price_range = sale_price_row["Units"].values[0]  # Example: "34900 - 755000"
        min_price, max_price = map(int, price_range.split(" - "))
        average_sale_price = (min_price + max_price) / 2
    else:
        st.error("SalePrice data is missing in the CSV.")
        return
    
    # Example placeholder values for other KPIs
    total_properties = 100  # Replace with actual calculation if available
    properties_sold = 80    # Replace with actual calculation if available
    average_days_on_market = 45  # Example placeholder
    st.write("""the Total properties, properties sold and average days on market
            are placeholders to be used if the dataset is expanded on!""")
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", total_properties)
    col2.metric("Avg. Sale Price ($)", f"{average_sale_price:,.2f}")
    col3.metric("Properties Sold", properties_sold)
    col4.metric("Avg. Days on Market", average_days_on_market)