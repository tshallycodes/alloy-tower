import pandas as pd
import numpy as np

# ─── LOAD ─────────────────────────────────────────────────────────
df = pd.read_csv('../data/raw/alloy_data.csv', sep=';')

# Fix BOM character in first column name
df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ─── 1. FIX COMMA-FORMATTED NUMERIC STRINGS ───────────────────────
# These columns contain numbers like "1,312" or "546,937.17"
comma_cols = {
    'sqft':             int,
    'lot_size_sqft':    int,
    'last_sale_price':  float,
    'days_since_sale':  int,
    'assessed_value':   float,
    'annual_tax':       float,
}

for col, dtype in comma_cols.items():
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype if col != 'last_sale_price' else float)

print("✓ Fixed comma-formatted numeric columns")

# ─── 2. PARSE DATE COLUMN ─────────────────────────────────────────
df['last_sale_date'] = pd.to_datetime(df['last_sale_date'], errors='coerce')
print("✓ Parsed last_sale_date to datetime")

# ─── 3. DROP REDUNDANT COLUMN ─────────────────────────────────────
# building_age = 2025 - year_built — redundant, keeping year_built
df.drop(columns=['building_age'], inplace=True)
print("✓ Dropped building_age (redundant with year_built)")

# ─── 4. STANDARDISE COLUMN NAME ───────────────────────────────────
df.rename(columns={'property_id': 'property_id'}, inplace=True)

# ─── 5. STRIP WHITESPACE FROM STRING COLUMNS ──────────────────────
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].str.strip()
print("✓ Stripped whitespace from string columns")

# ─── 6. UNIT COLUMN — fill nulls with empty string ────────────────
df['unit'] = df['unit'].fillna('')
print("✓ Filled unit nulls with empty string")

# ─── SAVE ─────────────────────────────────────────────────────────
output_path = '../data/raw/alloy_data_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Saved cleaned dataset to: {output_path}")