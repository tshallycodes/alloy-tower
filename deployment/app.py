import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --- Inline model loading (replaces API call for Streamlit Cloud deployment) ---
# import requests  # (API approach — commented out)
# API_URL = os.getenv("API_URL", "http://localhost:8000")  # (API approach — commented out)

PROPERTY_TYPES = ["Single Family", "Townhouse", "Condo", "Multi Family"]
STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]


def _find_file(filename: str, fallback_subdir: str):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(app_dir, filename),
        os.path.join(app_dir, '..', fallback_subdir, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


@st.cache_data
def load_zip_centroids() -> dict:
    return _find_file('zip_centroids.json', 'data')

@st.cache_data
def load_city_encoding() -> dict:
    return _find_file('city_encoding.json', 'models')

@st.cache_data
def load_state_encoding() -> dict:
    return _find_file('state_encoding.json', 'models')

@st.cache_data
def load_feature_columns() -> list:
    return _find_file('feature_columns.json', 'models')

@st.cache_data
def load_property_type_cols() -> list:
    return _find_file('property_type_cols.json', 'models')

@st.cache_resource
def load_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv("BEST_MODEL", "models/rf_model.pkl")
    candidates = [
        model_path,
        os.path.join(app_dir, '..', model_path),
        os.path.join(app_dir, model_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model not found. Tried: {candidates}")


# --- Load all assets ---
zip_centroids      = load_zip_centroids()
city_encoding      = load_city_encoding()
state_encoding     = load_state_encoding()
feature_columns    = load_feature_columns()
property_type_cols = load_property_type_cols()
model              = load_model()

CITIES = sorted(city_encoding.keys()) + ["Other"]

_CITY_MEAN  = sum(city_encoding.values()) / len(city_encoding) if city_encoding else 0.0
_STATE_MEAN = sum(state_encoding.values()) / len(state_encoding) if state_encoding else 0.0


def process_request(zip_code, latitude, longitude, bedrooms, bathrooms,
                    sqft, lot_size_sqft, year_built, days_since_sale,
                    tax_year, owner_occupied, property_type, city, state,
                    assessed_value, annual_tax) -> pd.DataFrame:
    row = {
        "zip_code":        zip_code,
        "latitude":        latitude,
        "longitude":       longitude,
        "bedrooms":        bedrooms,
        "bathrooms":       bathrooms,
        "sqft":            sqft,
        "lot_size_sqft":   lot_size_sqft,
        "year_built":      year_built,
        "days_since_sale": days_since_sale,
        "tax_year":        tax_year,
        "owner_occupied":  int(owner_occupied),
        "assessed_value":  np.log1p(assessed_value),
        "annual_tax":      annual_tax,
        "city":            city_encoding.get(city, _CITY_MEAN),
        "state":           state_encoding.get(state, _STATE_MEAN),
    }

    for col in property_type_cols:
        pt_value = col.replace("property_type_", "", 1)
        row[col] = 1 if property_type == pt_value else 0

    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


# ─── UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="AlloyTower Price Predictor", layout="centered")
st.title("AlloyTower Property Price Predictor")

# ─── Power BI Dashboard ───────────────────────────────────────────
st.divider()
st.subheader("📊 Data Analysis Dashboard")
st.write("Explore the interactive Power BI dashboard for deeper insights into the property data.")

POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiYmYwOTQwZDYtODc1ZC00NGIyLTk0NjgtYWNkYWQxNjY4MDEyIiwidCI6ImZmMGYzZTNhLTNlNTMtNDU0Zi1iMmI1LTZjNjg3NTNiOGVlNCJ9"


components.iframe(POWERBI_URL, width=None, height=600, scrolling=True)

st.caption(f"[Open dashboard in full screen ↗]({POWERBI_URL})")

# Add Space
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# ─── Input Form ───────────────────────────────────────────────────────
st.subheader("🏠 Property Details")
st.write("Enter property details to get an estimated sale price.")

# --- Coordinate session state ---
if 'latitude' not in st.session_state:
    st.session_state.latitude = 33.4484
if 'longitude' not in st.session_state:
    st.session_state.longitude = -112.074


def on_zip_change():
    zip_str = str(int(st.session_state.zip_widget))
    if zip_str in zip_centroids:
        c = zip_centroids[zip_str]
        st.session_state.latitude  = c['lat']
        st.session_state.longitude = c['lng']


# --- Location (outside form for live callback) ---
st.subheader("Location")
loc1, loc2, loc3 = st.columns(3)

with loc1:
    zip_code = st.number_input(
        "ZIP Code", min_value=10000, max_value=99999,
        value=85001, step=1, key='zip_widget', on_change=on_zip_change,
    )
with loc2:
    latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0,
        step=0.0001, format="%.4f", key='latitude',
    )
with loc3:
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0,
        step=0.0001, format="%.4f", key='longitude',
    )

maps_url = f"https://maps.google.com/?q={int(zip_code)}"
if str(int(zip_code)) in zip_centroids:
    st.caption(
        f"Coordinates auto-filled from ZIP {int(zip_code)} centroid. "
        f"Need a different spot? [Open in Google Maps]({maps_url}), "
        f"right-click the exact location, and copy the coordinates shown."
    )
else:
    st.caption(
        f"ZIP not in lookup. [Open ZIP {int(zip_code)} in Google Maps]({maps_url}), "
        f"right-click the location, and copy the coordinates into the fields above."
    )

st.divider()

# --- Property details form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        property_type  = st.selectbox("Property Type", PROPERTY_TYPES)
        city           = st.selectbox("City", CITIES, index=CITIES.index("Phoenix") if "Phoenix" in CITIES else 0)
        state          = st.selectbox("State", STATES, index=STATES.index("AZ"))
        bedrooms       = st.number_input("Bedrooms", min_value=1, max_value=20, value=3, step=1)
        bathrooms      = st.number_input("Bathrooms", min_value=1.0, max_value=20.0, value=2.0, step=0.5)
        assessed_value = st.number_input("Assessed Value ($)", min_value=0, max_value=10000000, value=300000, step=1000)
        owner_occupied = st.checkbox("Owner Occupied", value=True)


    with col2:
        sqft            = st.number_input("Square Feet", min_value=100, max_value=50000, value=1500, step=50)
        lot_size_sqft   = st.number_input("Lot Size (sqft)", min_value=0, max_value=500000, value=5000, step=100)
        year_built      = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990, step=1)
        days_since_sale = st.number_input("Days Since Last Sale", min_value=0, max_value=10000, value=365, step=1)
        tax_year        = st.number_input("Tax Year", min_value=2000, max_value=2025, value=2024, step=1)
        annual_tax      = st.number_input("Annual Tax ($)", min_value=0, max_value=200000, value=4000, step=100)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    if assessed_value > 1000000:
        st.warning("Predictions for properties above $1M assessed value may be less reliable.")
    with st.spinner("Predicting..."): 
        try:
            df = process_request(
                zip_code=int(zip_code),
                latitude=float(latitude),
                longitude=float(longitude),
                bedrooms=int(bedrooms),
                bathrooms=float(bathrooms),
                sqft=int(sqft),
                lot_size_sqft=float(lot_size_sqft),
                year_built=int(year_built),
                days_since_sale=int(days_since_sale),
                tax_year=int(tax_year),
                owner_occupied=owner_occupied,
                property_type=property_type,
                city=city,
                state=state,
                assessed_value=float(assessed_value),
                annual_tax=float(annual_tax),
            )
            price = float(model.predict(df)[0])
            st.success(f"Estimated Sale Price: **${price:,.2f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ─── ORIGINAL API-BASED CODE (commented out) ──────────────────────
#
# if submitted:
#     payload = {
#         "zip_code":       int(zip_code),
#         "latitude":       float(latitude),
#         "longitude":      float(longitude),
#         "bedrooms":       int(bedrooms),
#         "bathrooms":      float(bathrooms),
#         "sqft":           int(sqft),
#         "lot_size_sqft":  float(lot_size_sqft),
#         "year_built":     int(year_built),
#         "days_since_sale": int(days_since_sale),
#         "tax_year":       int(tax_year),
#         "owner_occupied": owner_occupied,
#         "property_type":  property_type,
#         "city":           city,
#         "state":          state,
#         "assessed_value": float(assessed_value),
#         "annual_tax":     float(annual_tax),
#     }
#
#     with st.spinner("Predicting..."):
#         try:
#             response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
#             if response.status_code == 200:
#                 price = response.json()["predicted_price"]
#                 st.success(f"Estimated Sale Price: **${price:,.2f}**")
#             else:
#                 st.error(f"API Error {response.status_code}: {response.text}")
#         except requests.exceptions.ConnectionError:
#             st.error(f"Cannot connect to API at {API_URL}. Is the API server running?")
#         except requests.exceptions.Timeout:
#             st.error("Request timed out. Please try again.")