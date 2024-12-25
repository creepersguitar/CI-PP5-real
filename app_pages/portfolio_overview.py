import streamlit as st
import pandas as pd
import plotly.express as px


# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_portfolio(data):
    st.header("Portfolio Overview")

    # Extract OverallQual descriptive row
    overall_qual_row = data[data["Variable"] == "OverallQual"]

    if not overall_qual_row.empty:
        # Extract the descriptive string from the "Units" column
        overall_qual_descriptions = overall_qual_row["Units"].values[0]
        
        # Parse the descriptions to extract valid numeric quality levels
        quality_levels = [int(entry.split(":")[0]) for entry in overall_qual_descriptions.split("; ")]
        min_qual, max_qual = min(quality_levels), max(quality_levels)
    else:
        st.error("OverallQual data is missing in the CSV.")
        return

    # Create a dummy property dataset (replace with real data if available)
    properties = {
        "Property ID": range(1, 11),  # Example property IDs
        "OverallQual": [min_qual, max_qual, 7, 6, 8, 5, 9, 4, 3, 10]  # Example quality levels
    }

    # Convert to DataFrame
    property_df = pd.DataFrame(properties)

    # Categorize properties based on OverallQual
    property_df["Property Type"] = property_df["OverallQual"].apply(
        lambda x: "High-end" if x >= 7 else "Standard"
    )

    # Display the updated DataFrame
    st.write(property_df)

    # Show distribution of property types
    st.subheader("Distribution of Property Types")
    st.bar_chart(property_df["Property Type"].value_counts())