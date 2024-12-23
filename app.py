import pandas as pd
import streamlit as st
from app_pages import dashboard, visualization, model_performance, bar


# Title for the app
st.title('House Price Prediction Dashboard')

# Sidebar for navigation
page = st.sidebar.radio("Select a Page", ["Home", "Data Visualizations", "Model Performance", "Predictions", "Bar chart"])

# Navigation logic
if page == "Home":
    st.write("Welcome to the House Price Prediction Dashboard. Here we explore and predict house sale prices.")
elif page == "Data Visualizations":
    visualization.show_visualizations()
elif page == "Model Performance":
    model_performance.show_metrics()
elif page == "Predictions":
    dashboard.make_predictions()
elif page == "Bar chart":
    bar.barchart()


st.set_page_config(
    page_title="Real Estate Dashboard",
    page_icon="🏠",
    layout="wide",
)
