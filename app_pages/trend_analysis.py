import streamlit as st
import plotly.express as px
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_trends(data):
    st.header("Trend Analysis")

    # Example: Average sale price over time (if you have a time column, e.g., 'YearBuilt')
    avg_price_by_year = data.groupby("YearBuilt")["SalePrice"].mean().reset_index()
    avg_price_by_year.columns = ["Year Built", "Avg Sale Price"]

    # Line chart for trends
    fig = px.line(
        avg_price_by_year,
        x="Year Built",
        y="Avg Sale Price",
        title="Average Sale Price Over Time",
        labels={"Avg Sale Price": "Average Price ($)", "Year Built": "Year"},
    )
    st.plotly_chart(fig)
