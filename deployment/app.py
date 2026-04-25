import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

PROPERTY_TYPES = ["Single Family", "Townhouse", "Condo", "Multi Family"]
STATES = [
    "AZ", "CA", "CO", "FL", "GA", "HI", "ID", "IL", "IN", "KS", "KY",
    "LA", "MA", "MD", "MI", "MN", "MO", "MT", "NC", "NE", "NM", "NV",
    "NY", "OH", "OK", "OR", "PA", "SC", "TN", "TX", "UT", "VA", "WA",
]

st.set_page_config(page_title="AlloyTower Price Predictor", layout="centered")
st.title("AlloyTower Property Price Predictor")
st.write("Enter property details to get an estimated sale price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        property_type = st.selectbox("Property Type", PROPERTY_TYPES)
        city = st.text_input("City", value="Phoenix")
        state = st.selectbox("State", STATES)
        zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999,
                                   value=85001, step=1)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20,
                                   value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=20.0,
                                    value=2.0, step=0.5)
        owner_occupied = st.checkbox("Owner Occupied", value=True)

    with col2:
        sqft = st.number_input("Square Feet", min_value=100, max_value=50000,
                               value=1500, step=50)
        lot_size_sqft = st.number_input("Lot Size (sqft)", min_value=0,
                                        max_value=500000, value=5000, step=100)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025,
                                     value=1990, step=1)
        days_since_sale = st.number_input("Days Since Last Sale", min_value=0,
                                          max_value=10000, value=365, step=1)
        tax_year = st.number_input("Tax Year", min_value=2000, max_value=2025,
                                   value=2024, step=1)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                   value=33.4484, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                    value=-112.074, step=0.0001, format="%.4f")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    payload = {
        "zip_code": int(zip_code),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft": int(sqft),
        "lot_size_sqft": float(lot_size_sqft),
        "year_built": int(year_built),
        "days_since_sale": int(days_since_sale),
        "tax_year": int(tax_year),
        "owner_occupied": owner_occupied,
        "property_type": property_type,
        "city": city,
        "state": state,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                price = response.json()["predicted_price"]
                st.success(f"Estimated Sale Price: **${price:,.2f}**")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Is the API server running?")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
