import streamlit as st
import pandas as pd
import plotly.express as px


# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def display_portfolio(data):
    st.header("Portfolio Overview")

    # Extract OverallQual range
    overall_qual_row = data[data["Variable"] == "OverallQual"]

    if not overall_qual_row.empty:
        # Parse the range from the "Units" column (e.g., "1 - 10")
        overall_qual_range = overall_qual_row["Units"].values[0]  # Example: "1 - 10"
        min_qual, max_qual = map(int, overall_qual_range.split(" - "))
    else:
        st.error("OverallQual data is missing in the CSV.")
        return

    # Create a dummy property dataset (replace this with real data)
    properties = {
        "Property ID": range(1, 11),  # Example property IDs
        "OverallQual": [min_qual, max_qual, 7, 6, 8, 5, 9, 4, 3, 10]  # Example qualities
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