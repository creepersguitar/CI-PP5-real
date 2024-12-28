import plotly.express as px
import streamlit as st
import pandas as pd
from app_pages.interactive_filters import apply_global_filters
fdata = apply_global_filters(data)

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

# Bar chart function
def barchart():
    # Filter for the 'SalePrice' row
    sale_price_row = data[data['Variable'] == 'SalePrice']

    # Check if 'SalePrice' exists in the dataset
    if not sale_price_row.empty:
        # Extract the units range
        units_range = sale_price_row['Units'].values[0]  # Get the 'Units' value
        units_values = list(map(int, units_range.split(" - ")))  # Convert range to a list of integers

        # Create a bar chart
        fig = px.bar(
            x=["Minimum", "Maximum"],  # Labels for the x-axis
            y=units_values,  # The min and max values from the range
            title='Sale Price Range',
            labels={'x': 'Range Type', 'y': 'Sale Price ($)'},
            text=units_values
        )

        # Customize the chart
        fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
    else:
        st.error("The 'SalePrice' variable is not found in the dataset.")

# Call the bar chart function
barchart()
