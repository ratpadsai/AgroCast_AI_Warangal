import streamlit as st
import numpy as np
import pickle

# Load model
model_data = pickle.load(open("model.pkl", "rb"))

rf_cotton = model_data["rf_cotton"]
rf_chili = model_data["rf_chili"]
scaler = model_data["scaler"]

st.title("🌾 AgroCast AI - Crop Price Prediction")

st.write("Predict next week's crop price for Warangal market")

# User Inputs
crop = st.selectbox("Select Crop", ["Cotton", "Chili"])
month = st.slider("Month", 1, 12)
temperature = st.number_input("Temperature (°C)", 15.0, 50.0, 30.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 200.0, 10.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0, 50.0)
prev_price = st.number_input("Previous Price (₹/quintal)", 1000.0, 30000.0, 6000.0)

if st.button("Predict Price"):

    # Feature engineering (same as notebook)
    year = 2025
    week = int(month * 4.33)
    quarter = (month - 1) // 3 + 1

    lag1 = prev_price
    lag4 = prev_price * 0.97
    ma4 = (lag1 + lag4) / 2

    features = np.array([[month, week, quarter, year,
                          temperature, rainfall, humidity,
                          lag1, lag4, ma4]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    if crop == "Cotton":
        prediction = rf_cotton.predict(features_scaled)[0]
    else:
        prediction = rf_chili.predict(features_scaled)[0]

    st.success(f"💰 Predicted Price: ₹ {round(prediction, 2)} per quintal")