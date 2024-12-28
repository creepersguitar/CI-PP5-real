import streamlit as st
import matplotlib.pyplot as plt
from app_pages.interactive_filters import apply_global_filters

def initialize(input_data):
    global fdata
    fdata = apply_global_filters(input_data)

def display_visualizations():
    # Use the filtered data here
    if fdata is not None:
        # Example: Generate a chart or visual
        print(fdata.head())
    else:
        print("Data not initialized.")

# Function to display visualizations
def show_visualizations():
    st.header("Data Visualizations")

    # Distribution of Sale Price (example)
    st.subheader("Sale Price Distribution")
    sale_prices = [34900, 455000, 300000, 200000, 400000]  # Replace with actual data
    plt.hist(sale_prices, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sale Prices')
    st.pyplot(plt)
