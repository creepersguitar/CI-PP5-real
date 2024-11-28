import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Example: Drop missing values
    df = df.dropna()
    return df
