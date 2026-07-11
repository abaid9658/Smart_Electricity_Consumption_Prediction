"""
feature_engineering.py
------------------------
Creates new engineered features from existing raw columns, WITHOUT using the
target variable (Daily_Electricity_Consumption_kWh) in any formula -- doing
so would cause DATA LEAKAGE (the model would implicitly "see" the answer
during training, giving unrealistically high accuracy that collapses on
real unseen data).

New features created:
  1. Total_Appliance_Hours   -> sum of all appliance usage hours
  2. Power_Usage_Index       -> weighted sum using approximate power ratings
                                 (a physics-informed proxy, NOT the target itself)
  3. Appliance_Density       -> appliance hours per sqft (usage intensity)
  4. Weather_Severity_Index  -> how far outdoor temp is from a comfortable 22C,
                                 combined with humidity stress
  5. Temperature_Category    -> binned temperature (Cold/Mild/Warm/Hot/Extreme)
  6. Occupancy_Score         -> how much the household is actually "at home"
                                 (family size x WFH x weekend x holiday)
  7. Appliance_Load_Score    -> count/weight of HIGH-power appliances in use
  8. Peak_Consumption_Flag_Score -> interacts Peak_Hour_Usage with AC/Geyser use
  9. Lifestyle_Category      -> categorical persona based on WFH + family size
"""

import pandas as pd
import numpy as np
import os

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_cleaned.csv")
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "electricity_consumption_raw.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_features.csv")

TARGET = "Daily_Electricity_Consumption_kWh"

# Same power ratings used in dataset generation -- reused here ONLY as domain
# knowledge to build a physics-informed index, not to reverse-engineer the target.
POWER_RATING = {
    "AC_Usage_Hours": 1.5, "Fan_Usage_Hours": 0.075, "TV_Hours": 0.10,
    "Laptop_Hours": 0.06, "Water_Motor_Hours": 0.75, "Microwave_Usage_Hours": 1.2,
    "Geyser_Usage_Hours": 2.0,
}


def engineer_features(df):
    df = df.copy()

    # 1. Total Appliance Hours -- simple sum of all usage-hour columns
    hour_cols = ["AC_Usage_Hours", "Fan_Usage_Hours", "TV_Hours", "Laptop_Hours",
                 "Water_Motor_Hours", "Microwave_Usage_Hours", "Geyser_Usage_Hours"]
    df["Total_Appliance_Hours"] = df[hour_cols].sum(axis=1)

    # 2. Power Usage Index -- weighted by real appliance wattage (physics-informed,
    #    but NOT the same formula as the target -- excludes lighting/EV/base load/
    #    solar adjustment/noise that the actual target has)
    df["Power_Usage_Index"] = sum(df[col] * weight for col, weight in POWER_RATING.items())

    # 3. Appliance Density -- usage intensity relative to house size
    df["Appliance_Density"] = df["Total_Appliance_Hours"] / df["Area_sqft"] * 1000

    # 4. Weather Severity Index -- distance from comfortable temp (22C) + humidity stress
    df["Weather_Severity_Index"] = (
        (df["Outdoor_Temperature"] - 22).abs() * 0.7 + (df["Humidity"] - 50).abs() * 0.1
    )

    # 5. Temperature Category (binned)
    def temp_bin(t):
        if t < 15: return "Cold"
        elif t < 25: return "Mild"
        elif t < 33: return "Warm"
        elif t < 40: return "Hot"
        else: return "Extreme"
    df["Temperature_Category"] = df["Outdoor_Temperature"].apply(temp_bin)

    # 6. Occupancy Score -- proxy for "how much is someone actually home today"
    df["Occupancy_Score"] = (
        df["Family_Members"] * (1 + 0.3 * df["Work_From_Home"] + 0.2 * df["Is_Weekend"] + 0.2 * df["Is_Holiday"])
    )

    # 7. Appliance Load Score -- weighted count of HIGH power appliances active
    df["Appliance_Load_Score"] = (
        (df["AC_Usage_Hours"] > 0).astype(int) * 3
        + (df["Geyser_Usage_Hours"] > 0).astype(int) * 2
        + df["EV_Charging"] * 3
        + df["Washing_Machine_Usage"] * 1
        + (df["Microwave_Usage_Hours"] > 0).astype(int) * 1
    )

    # 8. Peak Consumption interaction -- flags households whose heavy appliances
    #    (AC/Geyser) overlap with the reported peak usage window
    df["Peak_Load_Interaction"] = df["Peak_Hour_Usage"] * (df["AC_Usage_Hours"] + df["Geyser_Usage_Hours"])

    # 9. Lifestyle Category -- simple persona bucket from WFH + family size
    def lifestyle(row):
        if row["Work_From_Home"] == 1 and row["Family_Members"] >= 4:
            return "WFH_Large_Family"
        elif row["Work_From_Home"] == 1:
            return "WFH_Small_Household"
        elif row["Family_Members"] >= 5:
            return "Large_Family"
        elif row["Family_Members"] <= 2:
            return "Small_Household"
        else:
            return "Standard_Household"
    df["Lifestyle_Category"] = df.apply(lifestyle, axis=1)

    return df


if __name__ == "__main__":
    # Feature engineering works best on human-readable (unscaled, but already
    # cleaned/encoded) data -- we build on the cleaned dataset.
    df = pd.read_csv(CLEANED_PATH)
    df_fe = engineer_features(df)

    # One-hot encode the two NEW categorical columns we just created
    df_fe = pd.get_dummies(df_fe, columns=["Temperature_Category", "Lifestyle_Category"], drop_first=True)
    bool_cols = df_fe.select_dtypes(include="bool").columns
    df_fe[bool_cols] = df_fe[bool_cols].astype(int)

    df_fe.to_csv(OUT_PATH, index=False)

    print(f"New shape after feature engineering: {df_fe.shape}")
    new_cols = ["Total_Appliance_Hours", "Power_Usage_Index", "Appliance_Density",
                "Weather_Severity_Index", "Occupancy_Score", "Appliance_Load_Score",
                "Peak_Load_Interaction"]
    print("\nNew engineered feature stats:")
    print(df_fe[new_cols].describe().T[["mean", "std", "min", "max"]])

    print("\nCorrelation of NEW features with target:")
    print(df_fe[new_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False))

    print(f"\nSaved to: {OUT_PATH}")
