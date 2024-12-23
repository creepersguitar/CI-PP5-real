import streamlit as st
import pandas as pd
import plotly.express as px


# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_portfolio(data):
    st.header("Portfolio Overview")

    # Example: Property types based on 'OverallQual' column
    data["Property Type"] = data["OverallQual"].apply(lambda x: "High-end" if x >= 7 else "Standard")
    property_counts = data["Property Type"].value_counts().reset_index()
    property_counts.columns = ["Property Type", "Count"]

    # Create Pie Chart
    fig = px.pie(property_counts, values="Count", names="Property Type", title="Portfolio Distribution")
    st.plotly_chart(fig)