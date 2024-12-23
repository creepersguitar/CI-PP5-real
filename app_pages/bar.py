import plotly.express as px
import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

# Bar chart function without 'Region'
def barchart():
    # Check if the necessary column exists
    if 'Avg_Price' in data.columns:
        # Create a bar chart for average prices
        fig = px.bar(
            data,
            x=data.index,  # Use the index as the x-axis if no grouping column exists
            y='Avg_Price',
            title='Average Property Prices',
            labels={'Avg_Price': 'Average Price ($)', 'index': 'Property Index'},
            text='Avg_Price'
        )

        # Customize the chart
        fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            xaxis_tickangle=-45
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
    else:
        st.error("The column 'Avg_Price' is not found in the dataset.")

# Call the bar chart function
barchart()
