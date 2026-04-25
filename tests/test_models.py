import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def models_exist(*names: str) -> bool:
    return all(os.path.exists(os.path.join(MODELS_DIR, n)) for n in names)


class TestJSONArtifacts:
    def test_feature_columns_exists_and_is_list(self):
        path = os.path.join(MODELS_DIR, 'feature_columns.json')
        if not os.path.exists(path):
            pytest.skip("feature_columns.json not found — run feature_engineering.py first")
        with open(path) as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all(isinstance(c, str) for c in cols)

    def test_feature_columns_excludes_target(self):
        path = os.path.join(MODELS_DIR, 'feature_columns.json')
        if not os.path.exists(path):
            pytest.skip("feature_columns.json not found")
        with open(path) as f:
            cols = json.load(f)
        assert 'last_sale_price' not in cols

    def test_property_type_cols_prefixed(self):
        path = os.path.join(MODELS_DIR, 'property_type_cols.json')
        if not os.path.exists(path):
            pytest.skip("property_type_cols.json not found")
        with open(path) as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert all(c.startswith('property_type_') for c in cols)

    def test_city_encoding_is_dict_of_floats(self):
        path = os.path.join(MODELS_DIR, 'city_encoding.json')
        if not os.path.exists(path):
            pytest.skip("city_encoding.json not found")
        with open(path) as f:
            enc = json.load(f)
        assert isinstance(enc, dict)
        assert all(isinstance(v, (int, float)) for v in enc.values())

    def test_state_encoding_is_dict_of_floats(self):
        path = os.path.join(MODELS_DIR, 'state_encoding.json')
        if not os.path.exists(path):
            pytest.skip("state_encoding.json not found")
        with open(path) as f:
            enc = json.load(f)
        assert isinstance(enc, dict)
        assert len(enc) > 0


class TestModelLoading:
    def _load_feature_cols(self):
        path = os.path.join(MODELS_DIR, 'feature_columns.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _make_dummy_input(self, feature_cols):
        return pd.DataFrame([{col: 0.0 for col in feature_cols}])

    def test_gb_model_loads_and_predicts(self):
        path = os.path.join(MODELS_DIR, 'gb_model.pkl')
        if not os.path.exists(path):
            pytest.skip("gb_model.pkl not found — run model.ipynb first")
        model = joblib.load(path)
        assert hasattr(model, 'predict')
        feature_cols = self._load_feature_cols()
        if feature_cols:
            X = self._make_dummy_input(feature_cols)
            pred = model.predict(X)
            assert len(pred) == 1
            assert isinstance(pred[0], (int, float, np.floating))

    def test_rf_model_loads_and_predicts(self):
        path = os.path.join(MODELS_DIR, 'rf_model.pkl')
        if not os.path.exists(path):
            pytest.skip("rf_model.pkl not found — run model.ipynb first")
        model = joblib.load(path)
        assert hasattr(model, 'predict')
        feature_cols = self._load_feature_cols()
        if feature_cols:
            X = self._make_dummy_input(feature_cols)
            pred = model.predict(X)
            assert len(pred) == 1

    def test_ridge_model_loads_and_predicts(self):
        path = os.path.join(MODELS_DIR, 'ridge_model.pkl')
        if not os.path.exists(path):
            pytest.skip("ridge_model.pkl not found — run model.ipynb first")
        model = joblib.load(path)
        assert hasattr(model, 'predict')
        feature_cols = self._load_feature_cols()
        if feature_cols:
            X = self._make_dummy_input(feature_cols)
            pred = model.predict(X)
            assert len(pred) == 1

    def test_models_produce_positive_price(self):
        if not models_exist('gb_model.pkl', 'feature_columns.json'):
            pytest.skip("Models not found")
        model = joblib.load(os.path.join(MODELS_DIR, 'gb_model.pkl'))
        feature_cols = self._load_feature_cols()
        X = pd.DataFrame([{
            col: 1500 if 'sqft' in col else
                 3 if 'bedroom' in col else
                 2.0 if 'bathroom' in col else
                 1990 if 'year' in col else
                 1.0 if col.startswith('property_type_Single') else 0.0
            for col in feature_cols
        }])
        pred = model.predict(X)
        assert pred[0] > 0
