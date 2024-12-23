import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_kpis(data):
    st.header("Key Performance Indicators")

    # Calculate KPIs from the data
    total_properties = len(data)
    average_sale_price = data["SalePrice"].mean()
    properties_sold = data["SalePrice"].count()
    average_days_on_market = 45  # Example placeholder; replace if you have a relevant column

    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", total_properties)
    col2.metric("Avg. Sale Price ($)", f"{average_sale_price:,.2f}")
    col3.metric("Properties Sold", properties_sold)
    col4.metric("Avg. Days on Market", average_days_on_market)
