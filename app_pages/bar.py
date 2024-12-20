import plotly.express as px
import streamlit as st

def barchart():
    # Create the bar chart
    fig = px.bar(
        data,
        x='Region',
        y='Avg_Price',
        title='Average Property Prices by Region',
        labels={'Avg_Price': 'Average Price ($)', 'Region': 'Region'},
        color='Region',
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
