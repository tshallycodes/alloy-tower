import json
import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'alloy_data_cleaned.csv')
    out_path = os.path.join(base_dir, 'data', 'zip_centroids.json')

    df = pd.read_csv(raw_path, usecols=['zip_code', 'latitude', 'longitude'])
    df = df.dropna(subset=['zip_code', 'latitude', 'longitude'])
    df['zip_code'] = df['zip_code'].astype(int)

    centroids = (
        df.groupby('zip_code')[['latitude', 'longitude']]
        .median()
        .round(6)
        .rename(columns={'latitude': 'lat', 'longitude': 'lng'})
    )

    result = {
        str(zip_code): {'lat': float(row['lat']), 'lng': float(row['lng'])}
        for zip_code, row in centroids.iterrows()
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} ZIP centroids to {out_path}")


if __name__ == '__main__':
    main()
