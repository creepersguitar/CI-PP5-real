import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Example function to make predictions
def make_predictions():
    st.header("House Price Prediction")

    # Input form for prediction
    square_footage = st.number_input("Enter square footage of the house:", min_value=500, max_value=5000)
    num_bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10)
    garage_area = st.number_input("Enter garage area:", min_value=0, max_value=1000)

    # Predict using the model (Example model, replace with your trained model)
    model = RandomForestRegressor()
    model.fit([[1000, 3, 500], [1500, 4, 700]], [200000, 300000])  # Example data
    prediction = model.predict([[square_footage, num_bedrooms, garage_area]])

    # Display prediction
    if st.button("Predict"):
        st.write(f"The estimated sale price is: ${prediction[0]:,.2f}")
