import streamlit as st
import plotly.express as px
import pandas as pd

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_market_analysis(data):
    st.header("Market Analysis")

    # Calculate average sale price by 'OverallCond' (or another categorical column)
    avg_price_by_condition = data.groupby("OverallCond")["SalePrice"].mean().reset_index()
    avg_price_by_condition.columns = ["Overall Condition", "Avg Sale Price"]

    # Bar chart
    fig = px.bar(
        avg_price_by_condition,
        x="Overall Condition",
        y="Avg Sale Price",
        title="Average Sale Price by Overall Condition",
        labels={"Avg Sale Price": "Average Price ($)", "Overall Condition": "Condition Rating"},
    )
    st.plotly_chart(fig)
