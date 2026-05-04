import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AlloyTower Property Price Prediction API",
    description="Predict residential property sale prices using the best trained model.",
    version="1.0.0",
)

_MODEL_DIR = os.getenv("MODEL_DIR", "models")
_BEST_MODEL_PATH = os.getenv("BEST_MODEL", "models/rf_model.pkl")
_GLOBAL_MEAN = float(os.getenv("GLOBAL_PRICE_MEAN", "781993.0"))

model = joblib.load(_BEST_MODEL_PATH)

with open(os.path.join(_MODEL_DIR, "feature_columns.json")) as f:
    feature_columns: list[str] = json.load(f)

with open(os.path.join(_MODEL_DIR, "property_type_cols.json")) as f:
    property_type_cols: list[str] = json.load(f)

with open(os.path.join(_MODEL_DIR, "city_encoding.json")) as f:
    city_encoding: dict[str, float] = json.load(f)

with open(os.path.join(_MODEL_DIR, "state_encoding.json")) as f:
    state_encoding: dict[str, float] = json.load(f)

_CITY_MEAN = sum(city_encoding.values()) / len(city_encoding)
_STATE_MEAN = sum(state_encoding.values()) / len(state_encoding)


class PropertyInput(BaseModel):
    zip_code: int
    latitude: float
    longitude: float
    bedrooms: int
    bathrooms: float
    sqft: int
    lot_size_sqft: float
    year_built: int
    days_since_sale: int
    tax_year: int
    owner_occupied: bool
    property_type: str
    city: str
    state: str
    assessed_value: float
    annual_tax: float


def process_request(data: PropertyInput) -> pd.DataFrame:
    row: dict = {
        "zip_code": data.zip_code,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "sqft": data.sqft,
        "lot_size_sqft": data.lot_size_sqft,
        "year_built": data.year_built,
        "days_since_sale": data.days_since_sale,
        "tax_year": data.tax_year,
        "owner_occupied": int(data.owner_occupied),
        "city": city_encoding.get(data.city, _CITY_MEAN),
        "state": state_encoding.get(data.state, _STATE_MEAN),
        "assessed_value": np.log1p(data.assessed_value),
        "annual_tax": data.annual_tax,
    }

    for col in property_type_cols:
        pt_value = col.replace("property_type_", "", 1)
        row[col] = 1 if data.property_type == pt_value else 0

    df = pd.DataFrame([row])
    # Enforce column order; add missing columns as 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


@app.get("/")
def root():
    return {
        "message": "AlloyTower Property Price Prediction API",
        "model": _BEST_MODEL_PATH,
        "docs": "/docs",
    }


@app.post("/predict")
def predict(data: PropertyInput):
    df = process_request(data)
    prediction = float(model.predict(df)[0])
    return {"predicted_price": round(prediction, 2)}
