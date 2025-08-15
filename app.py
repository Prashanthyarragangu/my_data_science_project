import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved pipeline
model = joblib.load("best_model.pkl")

# Page title
st.title("Melbourne House Price Predictor")

# Sidebar inputs
st.sidebar.header("Enter Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
land_size = st.sidebar.number_input("Land Size (sqm)", min_value=0, value=300)
floor_area = st.sidebar.number_input("Floor Area (sqm)", min_value=0, value=120)
building_age = st.sidebar.slider("Building Age (years)", 0, 150, 30)
suburb = st.sidebar.text_input("Suburb", "Richmond")
property_type = st.sidebar.selectbox("Property Type", ['house', 'unit', 'townhouse'])
house_type = st.sidebar.selectbox("House Type", ['duplex', 'semi-detached', 'villa', 'standalone'])

# Predict button
if st.sidebar.button("Predict Price"):
    # Create DataFrame from input
    input_df = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'land_size_num': [land_size],
        'floor_area_num': [floor_area],
        'building_age': [building_age],
        'suburb': [suburb],
        'property_type': [property_type],
        'house_type': [house_type],
        'suburb_median_price': [0]  # placeholder if needed in pipeline
    })

    # Predict (make sure model handles log transformation internally)
    log_pred = model.predict(input_df)
    price_pred = np.expm1(log_pred)[0]

    st.subheader(f"Estimated Property Price: ${price_pred:,.0f}")
