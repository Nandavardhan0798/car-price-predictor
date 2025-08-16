# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

# Load the model
model = joblib.load("model.joblib")

# Patch function for monotonic_cst
def patch_monotonic(tree_model):
    if isinstance(tree_model, DecisionTreeRegressor) and not hasattr(tree_model, "monotonic_cst"):
        tree_model.monotonic_cst = None
    elif isinstance(tree_model, RandomForestRegressor):
        for est in tree_model.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
    elif isinstance(tree_model, Pipeline):
        patch_monotonic(tree_model.steps[-1][1])

patch_monotonic(model)

# Streamlit UI
st.title("Car Selling Price Predictor")
st.write("Enter car details below:")

# Example input fields (adjust to match your dataset)
brand = st.selectbox("Brand", ["Toyota", "Honda", "Maruti", "Hyundai", "Other"])
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("Mileage (km)", min_value=0, value=25000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])

# Add other columns required by your model
seats = st.number_input("Seats", min_value=2, max_value=12, value=5)
engine_cc = st.number_input("Engine CC", min_value=500, max_value=5000, value=1500)
torque_nm = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=150)
max_power_bhp = st.number_input("Max Power (BHP)", min_value=30, max_value=1000, value=100)
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
km_driven = st.number_input("KM Driven", min_value=0, value=25000)
mileage_mpg = st.number_input("Mileage (MPG)", min_value=5, max_value=100, value=20)

# Create input DataFrame
input_data = pd.DataFrame([{
    'owner': owner_type,
    'fuel': fuel_type,
    'seats': seats,
    'company': brand,
    'seller_type': seller_type,
    'engine_cc': engine_cc,
    'torque_nm': torque_nm,
    'max_power_bhp': max_power_bhp,
    'km_driven': km_driven,
    'mileage_mpg': mileage_mpg,
    'year': year,
    'transmission': transmission
}])

# Prediction button
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)
    st.success(f"Predicted Selling Price: ₹{predicted_price[0]:,.2f}")
