import streamlit as st
import plotly.express as px
import pandas as pd
from app_pages.interactive_filters import apply_global_filters

fdata = None

def initialize(input_data):
    global fdata
    fdata = apply_global_filters(input_data)

# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_market_analysis(data):
    st.header("Market Analysis")

    # Extract OverallCond descriptive row
    overall_cond_row = data[data["Variable"] == "OverallCond"]

    if not overall_cond_row.empty:
        # Parse the descriptions for conditions (e.g., "10: Very Excellent; 9: Excellent; ...")
        overall_cond_descriptions = overall_cond_row["Units"].values[0]
        
        # Extract numeric condition levels
        condition_levels = [int(entry.split(":")[0]) for entry in overall_cond_descriptions.split("; ")]
    else:
        st.error("OverallCond data is missing in the CSV.")
        return

    # Create a dummy dataset for analysis (replace with actual data)
    market_data = {
        "Property ID": range(1, 11),  # Example property IDs
        "OverallCond": [9, 7, 8, 6, 5, 10, 4, 3, 2, 1],  # Example condition levels
        "SalePrice": [450000, 350000, 400000, 320000, 300000, 500000, 280000, 200000, 150000, 100000]  # Example prices
    }

    # Convert to DataFrame
    market_df = pd.DataFrame(market_data)

    # Calculate average sale price by condition
    avg_price_by_condition = market_df.groupby("OverallCond")["SalePrice"].mean().reset_index()

    # Display the results
    st.subheader("Average Sale Price by Overall Condition")
    st.write(avg_price_by_condition)

    # Visualize the data
    st.bar_chart(avg_price_by_condition.set_index("OverallCond"))
    column = st.selectbox("Select Column for Outlier Detection", data.select_dtypes(include="number").columns)
    outliers = detect_outliers(data, column)
    st.write(outliers)

def detect_outliers(data, column, z_threshold=3):
    # Check if the column exists in the data
    if column not in data.columns:
        raise ValueError(f"The column '{column}' does not exist in the dataset.")
    
    # Calculate Z-scores for the specified column
    data["Z-Score"] = (data[column] - data[column].mean()) / data[column].std()
    
    # Detect outliers based on Z-score threshold
    outliers = data[np.abs(data["Z-Score"]) > z_threshold]
    
    # Plot the outliers with Plotly
    fig = px.scatter(data, x=column, y="Z-Score", color=np.abs(data["Z-Score"]) > z_threshold,
                     title=f"Outliers in {column}",
                     labels={"color": "Outlier"})
    st.plotly_chart(fig)
    
    # Display the number of outliers detected
    st.write(f"Number of Outliers in {column}: {len(outliers)}")
    
    return outliers