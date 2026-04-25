import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

VALID_PAYLOAD = {
    "zip_code": 85001,
    "latitude": 33.4484,
    "longitude": -112.074,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft": 1500,
    "lot_size_sqft": 5000.0,
    "year_built": 1990,
    "days_since_sale": 365,
    "tax_year": 2024,
    "owner_occupied": True,
    "property_type": "Single Family",
    "city": "Phoenix",
    "state": "AZ",
}


def models_ready() -> bool:
    required = [
        'gb_model.pkl', 'feature_columns.json', 'property_type_cols.json',
        'city_encoding.json', 'state_encoding.json',
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


@pytest.fixture(scope="module")
def client():
    if not models_ready():
        pytest.skip(
            "Models/JSON artefacts not found. "
            "Run feature_engineering.py and model.ipynb first."
        )
    os.environ.setdefault("BEST_MODEL", os.path.join(MODELS_DIR, "gb_model.pkl"))
    os.environ.setdefault("MODEL_DIR", MODELS_DIR)

    from fastapi.testclient import TestClient
    from deployment.main import app
    return TestClient(app)


class TestPredictEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_valid_input_returns_200_with_price(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], float)
        assert data["predicted_price"] > 0

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "sqft"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "bedrooms": "three"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_unknown_city_still_returns_prediction(self, client):
        payload = {**VALID_PAYLOAD, "city": "UnknownCityXYZ"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_price" in response.json()

    def test_unknown_state_still_returns_prediction(self, client):
        payload = {**VALID_PAYLOAD, "state": "ZZ"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
