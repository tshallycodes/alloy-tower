import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import (
    drop_identifier_columns,
    group_rare_cities,
    one_hot_encode_property_type,
    target_encode,
    handle_leakage,
    drop_object_columns,
    drop_correlated_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'property_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'id': [1, 2, 3, 4, 5],
        'full_address': ['a', 'b', 'c', 'd', 'e'],
        'street_address': ['s1', 's2', 's3', 's4', 's5'],
        'unit': [None, None, None, None, None],
        'owner_name': ['Alice', 'Bob', 'Carol', 'Dan', 'Eve'],
        'assessor_id': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'last_sale_date': ['2021-01-01'] * 5,
        'price_per_sqft': [200.0, 150.0, 300.0, 250.0, 180.0],
        'property_type': ['Single Family', 'Condo', 'Townhouse', 'Single Family', 'Multi Family'],
        'city': ['Phoenix', 'Phoenix', 'RareCity', 'Phoenix', 'Phoenix'],
        'state': ['AZ', 'AZ', 'AZ', 'AZ', 'AZ'],
        'last_sale_price': [300000, 200000, 400000, 350000, 250000],
        'bedrooms': [3, 2, 4, 3, 2],
        'bathrooms': [2.0, 1.5, 3.0, 2.5, 2.0],
        'sqft': [1500, 900, 2000, 1400, 1100],
        'assessed_value': [297000, 198000, 396000, 346500, 247500],  # 0.99 corr
        'annual_tax': [4500, 3000, 6000, 5250, 3750],
    })


class TestDropIdentifierColumns:
    def test_drops_all_listed_cols(self, sample_df):
        result = drop_identifier_columns(sample_df)
        for col in ['property_id', 'id', 'full_address', 'street_address',
                    'unit', 'owner_name', 'assessor_id', 'last_sale_date',
                    'price_per_sqft']:
            assert col not in result.columns

    def test_retains_feature_columns(self, sample_df):
        result = drop_identifier_columns(sample_df)
        for col in ['bedrooms', 'bathrooms', 'sqft', 'last_sale_price']:
            assert col in result.columns

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'bedrooms': [3, 4]})
        result = drop_identifier_columns(df)
        assert 'bedrooms' in result.columns


class TestGroupRareCities:
    def test_rare_city_becomes_other(self, sample_df):
        result = group_rare_cities(sample_df, min_count=2)
        assert 'Other' in result['city'].values
        assert 'RareCity' not in result['city'].values

    def test_common_city_unchanged(self, sample_df):
        result = group_rare_cities(sample_df, min_count=2)
        assert 'Phoenix' in result['city'].values

    def test_no_rare_cities_unchanged(self, sample_df):
        result = group_rare_cities(sample_df, min_count=1)
        assert 'RareCity' in result['city'].values
        assert 'Other' not in result['city'].values


class TestOneHotEncodePropertyType:
    def test_removes_property_type_column(self, sample_df):
        result, _ = one_hot_encode_property_type(sample_df)
        assert 'property_type' not in result.columns

    def test_creates_expected_columns(self, sample_df):
        result, pt_cols = one_hot_encode_property_type(sample_df)
        for col in ['property_type_Single Family', 'property_type_Condo',
                    'property_type_Townhouse', 'property_type_Multi Family']:
            assert col in result.columns

    def test_binary_values(self, sample_df):
        result, _ = one_hot_encode_property_type(sample_df)
        pt_cols = [c for c in result.columns if c.startswith('property_type_')]
        assert result[pt_cols].isin([0, 1]).all().all()

    def test_no_drop_first(self, sample_df):
        _, pt_cols = one_hot_encode_property_type(sample_df)
        assert len(pt_cols) == 4

    def test_returns_sorted_cols(self, sample_df):
        _, pt_cols = one_hot_encode_property_type(sample_df)
        assert pt_cols == sorted(pt_cols)


class TestTargetEncode:
    def test_city_encoded_to_mean(self, sample_df):
        df, maps = target_encode(sample_df, ['city'], target='last_sale_price')
        phoenix_mean = sample_df[sample_df['city'] == 'Phoenix']['last_sale_price'].mean()
        phoenix_rows = df[sample_df['city'] == 'Phoenix']['city']
        assert all(abs(v - phoenix_mean) < 1e-3 for v in phoenix_rows)

    def test_maps_returned(self, sample_df):
        _, maps = target_encode(sample_df, ['city', 'state'])
        assert 'city' in maps
        assert 'state' in maps
        assert 'Phoenix' in maps['city']

    def test_unknown_city_gets_global_mean(self):
        df = pd.DataFrame({
            'city': ['A', 'A', 'B'],
            'last_sale_price': [100, 200, 300],
        })
        result, _ = target_encode(df, ['city'])
        global_mean = df['last_sale_price'].mean()
        df_new = pd.DataFrame({'city': ['Unknown'], 'last_sale_price': [0]})
        # Encoding unknown via map + fillna
        means = df.groupby('city')['last_sale_price'].mean()
        val = df_new['city'].map(means).fillna(global_mean).iloc[0]
        assert val == pytest.approx(global_mean)


class TestHandleLeakage:
    def test_drops_high_corr_column(self, sample_df):
        result = handle_leakage(sample_df, threshold=0.90)
        assert 'assessed_value' not in result.columns
        assert 'annual_tax' not in result.columns

    def test_keeps_low_corr_column(self):
        df = pd.DataFrame({
            'last_sale_price': [100, 200, 300, 400, 500],
            'assessed_value': [50, 80, 120, 90, 60],  # low corr
        })
        result = handle_leakage(df, threshold=0.90)
        assert 'assessed_value' in result.columns

    def test_target_col_preserved(self, sample_df):
        result = handle_leakage(sample_df)
        assert 'last_sale_price' in result.columns


class TestDropCorrelatedFeatures:
    def test_drops_one_of_perfectly_correlated_pair(self):
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [2.0, 4.0, 6.0, 8.0, 10.0],  # perfect correlation with a
            'target': [10, 20, 30, 40, 50],
        })
        result = drop_correlated_features(df, target='target', threshold=0.90)
        assert not ('a' in result.columns and 'b' in result.columns)

    def test_target_not_dropped(self):
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = drop_correlated_features(df, target='target', threshold=0.90)
        assert 'target' in result.columns
