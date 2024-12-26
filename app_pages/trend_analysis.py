import streamlit as st
import plotly.express as px
import pandas as pd
from app_pages import apply_global_filters

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_trends(data):
    data = apply_global_filters(data)
    st.header("Trend Analysis")

    # Simulated data for trend analysis
    dummy_data = {
        "YearBuilt": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        "AvgSalePrice": [200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000, 300000]
    }
    trend_data = pd.DataFrame(dummy_data)

    # Line chart for average sale price by year
    fig = px.line(
        trend_data,
        x="YearBuilt",
        y="AvgSalePrice",
        title="Trend of Average Sale Price Over the Years",
        labels={"YearBuilt": "Year Built", "AvgSalePrice": "Average Sale Price ($)"}
    )

    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig)

    # Display simulated dataset
    st.subheader("Simulated Trend Data")
    st.write(trend_data)