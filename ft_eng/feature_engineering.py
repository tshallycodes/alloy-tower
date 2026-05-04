import os
import json
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'property_id', 'id', 'full_address', 'street_address', 'unit',
        'owner_name', 'assessor_id', 'last_sale_date', 'price_per_sqft',
    ]
    return df.drop(columns=[c for c in cols if c in df.columns])


def handle_leakage(df: pd.DataFrame, target: str = 'last_sale_price',
                   threshold: float = 0.90) -> pd.DataFrame:
    candidates = ['assessed_value', 'annual_tax']
    numeric = df.select_dtypes(include=[np.number])
    for col in candidates:
        if col not in numeric.columns:
            continue
        corr = abs(numeric[col].corr(numeric[target]))
        if corr >= threshold:
            print(f"  {col}: corr={corr:.4f} >= {threshold} -- DROPPED (circular leakage)")
            df = df.drop(columns=[col])
        else:
            print(f"  {col}: corr={corr:.4f} -- kept")
    return df


def group_rare_cities(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    counts = df['city'].value_counts()
    rare = counts[counts < min_count].index
    df = df.copy()
    df['city'] = df['city'].where(~df['city'].isin(rare), other='Other')
    if len(rare):
        print(f"  Grouped {len(rare)} rare cities into 'Other': {list(rare)}")
    else:
        print("  No cities with fewer than 10 properties; no grouping needed")
    return df


def one_hot_encode_property_type(df: pd.DataFrame) -> tuple:
    dummies = pd.get_dummies(df['property_type'], prefix='property_type',
                             drop_first=False).astype(int)
    cols = sorted(dummies.columns.tolist())
    dummies = dummies[cols]
    df = pd.concat([df.drop(columns=['property_type']), dummies], axis=1)
    return df, cols


def target_encode(df: pd.DataFrame, cols: list,
                  target: str = 'last_sale_price') -> tuple:
    df = df.copy()
    global_mean = df[target].mean()
    maps = {}
    for col in cols:
        means = df.groupby(col)[target].mean()
        maps[col] = means.to_dict()
        df[col] = df[col].map(means).fillna(global_mean)
    return df, maps


def drop_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"  Dropping remaining object columns: {obj_cols}")
        df = df.drop(columns=obj_cols)
    return df


def drop_correlated_features(df: pd.DataFrame, target: str = 'last_sale_price',
                              threshold: float = 0.90,
                              protected: list = None) -> pd.DataFrame:
    protected_set = set(protected or [])
    feature_cols = [c for c in df.select_dtypes(include=[np.number, 'bool']).columns
                    if c != target]
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns
               if c not in protected_set and (upper[c] > threshold).any()]
    if to_drop:
        print(f"  Dropping correlated features (>{threshold}): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print(f"  No feature pairs exceed the {threshold} correlation threshold")
    return df


def run() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'alloy_data_cleaned.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = load_data(raw_path)
    print(f"Loaded: {df.shape}")

    print("Dropping identifier / direct-leakage columns...")
    df = drop_identifier_columns(df)

    print("Grouping rare cities (<10 properties)...")
    df = group_rare_cities(df)

    print("One-hot encoding property_type...")
    df, property_type_cols = one_hot_encode_property_type(df)

    print("Target encoding city and state...")
    df, encoding_maps = target_encode(df, ['city', 'state'])

    print("Dropping remaining object columns...")
    df = drop_object_columns(df)

    # Cast bool owner_occupied to int for model compatibility
    if 'owner_occupied' in df.columns:
        df['owner_occupied'] = df['owner_occupied'].astype(int)

    print("Dropping highly correlated features (threshold=0.90)...")
    df = drop_correlated_features(df, protected=['city', 'state', 'assessed_value', 'annual_tax'])

    out_path = os.path.join(processed_dir, 'ft_eng.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved processed data: {out_path}  shape={df.shape}")

    feature_cols = [c for c in df.columns if c != 'last_sale_price']

    with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(models_dir, 'property_type_cols.json'), 'w') as f:
        json.dump(property_type_cols, f, indent=2)

    with open(os.path.join(models_dir, 'city_encoding.json'), 'w') as f:
        json.dump(encoding_maps['city'], f, indent=2)

    with open(os.path.join(models_dir, 'state_encoding.json'), 'w') as f:
        json.dump(encoding_maps['state'], f, indent=2)

    print(f"Saved JSON artefacts to models/")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return df


if __name__ == '__main__':
    run()
