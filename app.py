import pandas as pd
import streamlit as st
from app_pages import dashboard, visualization, model_performance, bar,kpis, portfolio_overview, market_analysis, interactive_filters, alerts, trend_analysis, search

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')


st.set_page_config(
    page_title="Real Estate Dashboard",
    page_icon="🏠",
    layout="wide",
)

# Title for the app
st.title('House Price Prediction Dashboard')

# Sidebar for navigation
page = st.sidebar.radio("Select a Page", ["Home", "Data Visualizations", "Model Performance", "Predictions", "Bar chart", "KPIs", "Portfolio Overview", "Market Analysis", 
                                  "Interactive Filters", "Alerts", "Trend Analysis"])

# Navigation logic
if page == "Home":
    st.write("Welcome to the House Price Prediction Dashboard. Here we explore and predict house sale prices.")
elif page == "Data Visualizations":
    visualization.show_visualizations()
elif page == "Model Performance":
    model_performance.show_metrics()
elif page == "Predictions":
    dashboard.make_predictions(data)
elif page == "Bar chart":
    bar.barchart()
elif page == "KPIs":
    kpis.display_kpis(data)
elif page == "Portfolio Overview":
    portfolio_overview.display_portfolio(data)
elif page == "Market Analysis":
    market_analysis.display_market_analysis(data)
elif page == "Interactive Filters":
    interactive_filters.display_filters(data)
elif page == "Alerts":
    alerts.display_alerts(data)
elif page == "Trend Analysis":
    trend_analysis.display_trends(data)
elif page == "Search":
    search_properties(data)