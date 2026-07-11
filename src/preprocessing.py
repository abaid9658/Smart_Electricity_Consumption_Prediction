"""
preprocessing.py
-----------------
Cleans the raw dataset and prepares it for machine learning.

Steps performed (in order, because order matters):
  1. Missing value handling  -> median imputation (robust to outliers)
  2. Duplicate removal       -> drop exact duplicate rows
  3. Outlier detection       -> IQR method, cap (winsorize) rather than delete
                                 (deleting real rows loses information; capping
                                 keeps the row but limits extreme influence)
  4. Feature encoding        -> One-Hot Encoding for nominal categoricals
  5. Feature scaling         -> StandardScaler (mean=0, std=1) for numeric columns
  6. Save processed dataset + fitted scaler (needed later for the prediction app)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "electricity_consumption_raw.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TARGET = "Daily_Electricity_Consumption_kWh"
CATEGORICAL_COLS = ["House_Type", "Season", "Day_of_Week"]
ID_COLS = ["House_ID"]  # identifiers -- never used as ML features

# IQR-based outlier detection only makes sense for genuinely CONTINUOUS columns.
# Binary flags (0/1) and near-constant columns will falsely trigger "outliers"
# on their minority class -- e.g. Solar_Panel=1 (only 15% of houses) would get
# flagged and capped, which is meaningless since there's no such thing as an
# "outlier" in a yes/no flag.
CONTINUOUS_COLS_FOR_OUTLIERS = [
    "Area_sqft", "Number_of_Rooms", "Family_Members", "Adults", "Children",
    "AC_Usage_Hours", "Fan_Usage_Hours", "TV_Hours", "Laptop_Hours",
    "Lighting_Hours_Per_Room", "Water_Motor_Hours", "Microwave_Usage_Hours",
    "Geyser_Usage_Hours", "Outdoor_Temperature", "Humidity",
    "Electricity_Rate", TARGET,
]


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def handle_missing_values(df):
    """Median imputation for numeric columns. Median chosen over mean because
    it is robust to skew/outliers -- a few extreme values won't drag it around."""
    before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    after = df.isnull().sum().sum()
    print(f"[Missing Values] Filled {before - after} missing cells using median imputation.")
    return df


def remove_duplicates(df):
    before = len(df)
    # Drop duplicates ignoring House_ID (since duplicated rows kept the same ID)
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ID_COLS])
    after = len(df)
    print(f"[Duplicates] Removed {before - after} duplicate rows. New shape: {df.shape}")
    return df


def handle_outliers(df, cols=None):
    """IQR-based capping (winsorization). We CAP instead of DELETE because:
    - Deleting rows loses valid information from other columns in that row
    - Capping keeps the row but limits the extreme value's influence on the model
    """
    if cols is None:
        cols = [c for c in CONTINUOUS_COLS_FOR_OUTLIERS if c in df.columns]

    capped_counts = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            capped_counts[col] = int(n_outliers)
            df[col] = df[col].clip(lower=lower, upper=upper)

    print(f"[Outliers] Capped values in {len(capped_counts)} columns: {capped_counts}")
    return df


def encode_features(df):
    """One-Hot Encoding for nominal categorical columns (no natural order).
    drop_first=True avoids the 'dummy variable trap' (multicollinearity) --
    e.g. if we know it's not Summer/Winter/Spring, it must be Autumn, so we
    don't need a separate Autumn column."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    # Convert boolean dummy columns to int (0/1) for consistency with ML libraries
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    print(f"[Encoding] One-Hot Encoded columns: {CATEGORICAL_COLS}")
    print(f"[Encoding] Shape after encoding: {df_encoded.shape}")
    return df_encoded


def scale_features(df, target=TARGET, id_cols=ID_COLS):
    """StandardScaler transforms each feature to mean=0, std=1.
    WHY: Algorithms like Linear Regression, SVR, and distance-based models
    are sensitive to feature magnitude -- without scaling, a feature like
    Area_sqft (~900-2800) would dominate a feature like AC_Usage_Hours (~0-16)
    purely because of its larger numeric range, not because it's more important.
    Tree-based models (Random Forest, XGBoost) don't need this, but we prepare
    a scaled version so ALL model types in Step 6 can use the same pipeline."""
    feature_cols = [c for c in df.columns if c not in id_cols + [target]]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"[Scaling] StandardScaler fitted on {len(feature_cols)} numeric feature columns.")
    return df_scaled, scaler


def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data()
    print(f"[Load] Raw data shape: {df.shape}\n")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df_encoded = encode_features(df)

    # Save the CLEANED (unscaled) version -- useful for EDA and tree-based models
    cleaned_path = os.path.join(PROCESSED_DIR, "electricity_cleaned.csv")
    df_encoded.to_csv(cleaned_path, index=False)
    print(f"\n[Save] Cleaned (unscaled) dataset saved to: {cleaned_path}")

    # Save the SCALED version -- useful for Linear Regression / SVR
    df_scaled, scaler = scale_features(df_encoded)
    scaled_path = os.path.join(PROCESSED_DIR, "electricity_scaled.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"[Save] Scaled dataset saved to: {scaled_path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Save] Fitted scaler saved to: {scaler_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final cleaned shape: {df_encoded.shape}")
    print(f"Final scaled shape:  {df_scaled.shape}")

    return df_encoded, df_scaled, scaler


if __name__ == "__main__":
    run_pipeline()
