"""
predict.py
----------
Unified prediction pipeline: takes raw user input (the same kind of fields
collected in the original dataset), engineers the same features used in
training, aligns columns to the exact order the model expects, and returns
a prediction + full recommendation summary.

This mirrors, in miniature, the exact transformations done in:
  preprocessing.py -> feature_engineering.py -> train.py
It is critical that this pipeline stays IN SYNC with those scripts, since a
mismatch (e.g. different column order or missing dummy column) is one of
the most common real-world bugs in deployed ML systems.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from recommendation_engine import generate_recommendations

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

POWER_RATING = {
    "AC_Usage_Hours": 1.5, "Fan_Usage_Hours": 0.075, "TV_Hours": 0.10,
    "Laptop_Hours": 0.06, "Water_Motor_Hours": 0.75, "Microwave_Usage_Hours": 1.2,
    "Geyser_Usage_Hours": 2.0,
}


def _temp_bin(t):
    if t < 15: return "Cold"
    elif t < 25: return "Mild"
    elif t < 33: return "Warm"
    elif t < 40: return "Hot"
    else: return "Extreme"


def _lifestyle(work_from_home, family_members):
    if work_from_home == 1 and family_members >= 4:
        return "WFH_Large_Family"
    elif work_from_home == 1:
        return "WFH_Small_Household"
    elif family_members >= 5:
        return "Large_Family"
    elif family_members <= 2:
        return "Small_Household"
    else:
        return "Standard_Household"


def build_feature_row(raw: dict, feature_cols: list) -> pd.DataFrame:
    """Takes a dict of raw user inputs and builds a single-row DataFrame with
    ALL engineered features + one-hot columns, matching the training schema
    exactly (missing dummy columns are filled with 0)."""

    row = dict(raw)  # copy

    hour_cols = ["AC_Usage_Hours", "Fan_Usage_Hours", "TV_Hours", "Laptop_Hours",
                 "Water_Motor_Hours", "Microwave_Usage_Hours", "Geyser_Usage_Hours"]
    row["Total_Appliance_Hours"] = sum(row.get(c, 0) for c in hour_cols)
    row["Power_Usage_Index"] = sum(row.get(c, 0) * w for c, w in POWER_RATING.items())
    row["Appliance_Density"] = row["Total_Appliance_Hours"] / max(row.get("Area_sqft", 1000), 1) * 1000
    row["Weather_Severity_Index"] = (
        abs(row.get("Outdoor_Temperature", 25) - 22) * 0.7 + abs(row.get("Humidity", 50) - 50) * 0.1
    )
    row["Occupancy_Score"] = row.get("Family_Members", 1) * (
        1 + 0.3 * row.get("Work_From_Home", 0) + 0.2 * row.get("Is_Weekend", 0) + 0.2 * row.get("Is_Holiday", 0)
    )
    row["Appliance_Load_Score"] = (
        (row.get("AC_Usage_Hours", 0) > 0) * 3
        + (row.get("Geyser_Usage_Hours", 0) > 0) * 2
        + row.get("EV_Charging", 0) * 3
        + row.get("Washing_Machine_Usage", 0) * 1
        + (row.get("Microwave_Usage_Hours", 0) > 0) * 1
    )
    row["Peak_Load_Interaction"] = row.get("Peak_Hour_Usage", 0) * (
        row.get("AC_Usage_Hours", 0) + row.get("Geyser_Usage_Hours", 0)
    )

    temp_cat = _temp_bin(row.get("Outdoor_Temperature", 25))
    lifestyle_cat = _lifestyle(row.get("Work_From_Home", 0), row.get("Family_Members", 1))

    # Build a single-row DataFrame, then apply the same one-hot pattern via reindex
    df_row = pd.DataFrame([row])

    # Manually create the one-hot flags this row would have produced
    for cat_val in ["Cold", "Mild", "Warm", "Hot", "Extreme"]:
        col = f"Temperature_Category_{cat_val}"
        if col in feature_cols:
            df_row[col] = 1 if temp_cat == cat_val else 0

    for cat_val in ["Large_Family", "Small_Household", "Standard_Household",
                     "WFH_Large_Family", "WFH_Small_Household"]:
        col = f"Lifestyle_Category_{cat_val}"
        if col in feature_cols:
            df_row[col] = 1 if lifestyle_cat == cat_val else 0

    for cat_val in ["Independent House", "Villa"]:
        col = f"House_Type_{cat_val}"
        if col in feature_cols:
            df_row[col] = 1 if row.get("House_Type") == cat_val else 0

    for cat_val in ["Spring", "Summer", "Winter"]:
        col = f"Season_{cat_val}"
        if col in feature_cols:
            df_row[col] = 1 if row.get("Season") == cat_val else 0

    for cat_val in ["Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]:
        col = f"Day_of_Week_{cat_val}"
        if col in feature_cols:
            df_row[col] = 1 if row.get("Day_of_Week") == cat_val else 0

    # Reindex to EXACT training column order; any column not set defaults to 0
    df_row = df_row.reindex(columns=feature_cols, fill_value=0)
    return df_row


def predict_consumption(raw_input: dict):
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

    X_row = build_feature_row(raw_input, feature_cols)
    predicted_kwh = float(model.predict(X_row)[0])

    electricity_rate = raw_input.get("Electricity_Rate", 22.0)
    summary, recommendations = generate_recommendations(raw_input, predicted_kwh, electricity_rate)

    return {
        "predicted_kwh": round(predicted_kwh, 2),
        "summary": summary,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    sample = {
        "House_Type": "Independent House", "Area_sqft": 1500, "Number_of_Rooms": 4,
        "Family_Members": 4, "Adults": 2, "Children": 2, "Senior_Citizens": 0,
        "Work_From_Home": 1, "AC_Usage_Hours": 8, "Fan_Usage_Hours": 6,
        "TV_Hours": 4, "Laptop_Hours": 5, "Lighting_Hours_Per_Room": 6,
        "Washing_Machine_Usage": 1, "Water_Motor_Hours": 1, "Microwave_Usage_Hours": 0.3,
        "Geyser_Usage_Hours": 1.5, "EV_Charging": 0, "Solar_Panel": 0, "Battery_Backup": 0,
        "Outdoor_Temperature": 35, "Humidity": 60, "Season": "Summer", "Month": 5,
        "Day_of_Week": "Monday", "Is_Weekend": 0, "Is_Holiday": 0, "Peak_Hour_Usage": 1,
        "Electricity_Rate": 22.0,
    }
    result = predict_consumption(sample)
    print("Predicted kWh:", result["predicted_kwh"])
    print("\nSummary:", result["summary"])
    print("\nRecommendations:")
    for r in result["recommendations"]:
        print(f"- [{r['category']}] {r['message']}")
