import streamlit as st
import pandas as pd
# Load the dataset
data = pd.read_csv('assets/AmesHousing.csv')

def search_properties(data):
    search_term = st.sidebar.text_input("Search for a Property (Keyword)", "")
    
    if search_term:
        filtered_data = data[
            data.apply(lambda row: search_term.lower() in row.to_string().lower(), axis=1)
        ]
        st.write(f"Found {len(filtered_data)} matching properties.")
        return filtered_data
    else:
        return data
