import streamlit as st
import pandas as pd
from app_pages.interactive_filters import apply_global_filters

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_alerts(data):
    data = apply_global_filters(data)
    st.header("Property Alerts")

    # Simulated dataset for testing
    dummy_data = {
        "Property ID": range(1, 11),
        "OverallQual": [7, 8, 6, 5, 9, 8, 4, 3, 10, 2],
        "SalePrice": [450000, 500000, 350000, 300000, 600000, 480000, 280000, 200000, 750000, 150000],
        "GrLivArea": [2000, 2500, 1800, 1600, 3000, 2400, 1400, 1200, 3500, 1000],
    }
    alert_data = pd.DataFrame(dummy_data)

    # Set alert thresholds
    expensive_threshold = 500000
    small_area_threshold = 1500

    # Properties above expensive threshold
    expensive_properties = alert_data[alert_data["SalePrice"] > expensive_threshold]
    if not expensive_properties.empty:
        st.subheader("Expensive Properties Alert")
        st.write("The following properties exceed the sale price threshold:")
        st.write(expensive_properties)
    else:
        st.subheader("Expensive Properties Alert")
        st.write("No properties exceed the sale price threshold.")

    # Properties with small living area
    small_properties = alert_data[alert_data["GrLivArea"] < small_area_threshold]
    if not small_properties.empty:
        st.subheader("Small Living Area Alert")
        st.write("The following properties have a smaller living area than the threshold:")
        st.write(small_properties)
    else:
        st.subheader("Small Living Area Alert")
        st.write("No properties have a smaller living area than the threshold.")