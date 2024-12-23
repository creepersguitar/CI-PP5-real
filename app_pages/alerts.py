import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_alerts(data):
    st.header("Alerts and Notifications")

    # Example alert for high inventory
    total_inventory = len(data)
    if total_inventory > 1000:
        st.warning("High inventory levels detected! Consider adjusting property prices.")

    # Alert for properties with very low or very high prices
    expensive_properties = data[data["SalePrice"] > 500000]
    cheap_properties = data[data["SalePrice"] < 100000]

    if not expensive_properties.empty:
        st.info(f"There are {len(expensive_properties)} high-end properties priced above $500,000.")
    if not cheap_properties.empty:
        st.info(f"There are {len(cheap_properties)} budget properties priced below $100,000.")
