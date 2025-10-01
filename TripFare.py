import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from geopy.distance import geodesic

#9 📌 Final Task: Build a Streamlit UI
#After training and evaluating your regression models:
#● Create a Streamlit UI where users can input relevant trip details such as pickup
# and dropoff locations, passenger count, time of travel, and other trip-related features.
#● On submitting the inputs: Display the predicted total fare amount using your
# best regression model.

model = joblib.load("gradient_boosting.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚖 Taxi Fare Prediction App")
st.write("**📝 Enter your trip details to get an 💰 Estimated Fare Amount :**")

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.write(
        """
        🚖 **Taxi Fare Prediction App**  
        This app estimates taxi fares based on:  

        - 📏 Trip Distance  
        - ⏱️ Trip Duration  
        - 🏢 Vendor Info  
        - 💳 Payment Type  
        - 🌙 Day/Night Timing  
        """
    )

st.markdown(
    "<p style='font-size:16px; font-weight:bold; margin-bottom:0px;'>How would you like to provide your trip details?</p>",
    unsafe_allow_html=True)
input_method = st.radio("", ["📍 Pickup & Dropoff Coordinates and Time", "📏 Distance and Duration"])

trip_duration = None
trip_distance = None
is_night = None

if input_method == "📍 Pickup & Dropoff Coordinates and Time":

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📍 Pickup Location**")
        pickup_lat = st.number_input("Pickup Latitude", -90.0, 90.0, step=0.0001, format="%.4f", key="pickup_lat")
        pickup_lon = st.number_input("Pickup Longitude", -180.0, 180.0, step=0.0001, format="%.4f", key="pickup_lon")

    with col2:
        st.markdown("**📍 Dropoff Location**")
        dropoff_lat = st.number_input("Dropoff Latitude", -90.0, 90.0, step=0.0001, format="%.4f", key="dropoff_lat")
        dropoff_lon = st.number_input("Dropoff Longitude", -180.0, 180.0, step=0.0001, format="%.4f", key="dropoff_lon")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🕒 Pickup Time**")
        pickup_date = st.date_input("Pickup Date", datetime.date.today(), key="pickup_date")
        pickup_time = st.time_input("Pickup Time", datetime.time(9, 0), key="pickup_time")

    with col4:
        st.markdown("**🕒 Dropoff Time**")
        dropoff_date = st.date_input("Dropoff Date", datetime.date.today(), key="dropoff_date")
        dropoff_time = st.time_input("Dropoff Time", datetime.time(9, 30), key="dropoff_time")

    pickup_dt = datetime.datetime.combine(pickup_date, pickup_time)
    dropoff_dt = datetime.datetime.combine(dropoff_date, dropoff_time)

    if dropoff_dt <= pickup_dt:
        st.error("❌ Dropoff time must be after Pickup time. Please correct the times.")
    else:
        if pickup_lat and pickup_lon and dropoff_lat and dropoff_lon:
            trip_distance = geodesic((pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)).km
        else:
            trip_distance = 0.0

        trip_duration = (dropoff_dt - pickup_dt).total_seconds() / 60.0
        st.write(f"⏱️ Trip Duration: {trip_duration:.1f} Minutes")
        st.write(f"📏 Trip Distance: {trip_distance:.2f} Km")

        pickup_hour = pickup_dt.hour
        is_night = 1 if (pickup_hour >= 22 or pickup_hour < 5) else 0
        st.write("🌙 Trip Type:", "Night Trip" if is_night else "Day Trip")

else:
    trip_distance = st.number_input("📏 Trip Distance (in Km)", min_value=0.1, max_value=500.0, step=0.1)
    trip_duration = st.number_input("⏱️ Trip Duration (in Minutes)", min_value=1, max_value=300, step=1)
    is_night_select = st.selectbox("🌙 Is Night Trip?", ["No", "Yes"])
    is_night = 1 if is_night_select == "Yes" else 0

vendor = st.selectbox("🏢 Vendor ID", [1, 2])
ratecode = st.selectbox("🏷️ Ratecode ID", [1, 2, 3, 4, 5, 6, 99])
payment_type = st.selectbox("💳 Payment Type", [1, 2, 3, 4])

if st.button("Predict Fare"):
    if trip_duration is None or trip_distance is None or is_night is None:
        st.warning("⚠️ Please ensure valid trip details and times before predicting.")
    else:
        num_features = pd.DataFrame([{
            "trip_distance": np.log1p(trip_distance),
            "trip_duration_min": np.log1p(trip_duration)
        }])

        num_scaled = scaler.transform(num_features)
        trip_distance_scaled, trip_duration_scaled = num_scaled[0]

        vendor2 = 1 if vendor == 2 else 0
        rate2 = 1 if ratecode == 2 else 0
        rate3 = 1 if ratecode == 3 else 0
        rate4 = 1 if ratecode == 4 else 0
        pay2 = 1 if payment_type == 2 else 0
        pay3 = 1 if payment_type == 3 else 0
        pay4 = 1 if payment_type == 4 else 0

        X_final = pd.DataFrame([[trip_distance_scaled, is_night, trip_duration_scaled,
            vendor2, rate2, rate3, rate4, pay2, pay3, pay4]], columns=[
            'trip_distance', 'is_night', 'trip_duration_min',
            'VendorID_2', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4',
            'payment_type_2', 'payment_type_3', 'payment_type_4'])

        fare_pred = model.predict(X_final)[0]
        st.success(f"Predicted Fare: $ {fare_pred:.2f}")
