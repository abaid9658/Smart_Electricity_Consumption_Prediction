"""
generate_dataset.py
--------------------
Generates a REALISTIC synthetic household electricity consumption dataset.

WHY SYNTHETIC DATA IS DESIGNED THIS WAY:
Real electricity consumption follows physics: Energy (kWh) = Power (kW) x Time (hours).
So instead of pure random numbers, we:
  1. Assign each appliance an approximate real-world power rating (kW).
  2. Generate realistic usage-hours per appliance, influenced by logical factors
     (temperature -> AC hours, family size -> appliance usage, weekend -> more usage, etc.)
  3. Multiply power x hours to get energy contribution per appliance.
  4. Sum all appliance contributions + base load (standby devices) = total daily kWh.
  5. Add small random noise to mimic real-world unpredictability (e.g. voltage
     fluctuation, appliance efficiency differences).

This ensures our target variable (Daily_Electricity_Consumption_kWh) is
STATISTICALLY LEARNABLE from the features -- exactly what a real dataset would be.
"""

import numpy as np
import pandas as pd
import os

# Reproducibility: same seed = same dataset every time we run this script.
# This is CRITICAL for ML projects so results can be reproduced/verified.
np.random.seed(42)

N_RECORDS = 2000  # more than the required 500, for better model performance

# -------------------------------------------------------------------
# Approximate real-world appliance power ratings in kW (industry avg values)
# -------------------------------------------------------------------
POWER_RATING = {
    "AC": 1.5,
    "Fan": 0.075,
    "Refrigerator": 0.15,
    "TV": 0.10,
    "Laptop": 0.06,
    "Lighting_Per_Room": 0.06,
    "Washing_Machine": 0.50,   # per use (not per hour)
    "Water_Motor": 0.75,
    "Microwave": 1.20,
    "Geyser": 2.00,
    "EV_Charging": 7.00,       # if charged that day
}

SEASONS = ["Summer", "Winter", "Spring", "Autumn"]
HOUSE_TYPES = ["Apartment", "Independent House", "Villa"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_dataset(n=N_RECORDS):
    rows = []

    for i in range(1, n + 1):
        # ---------------- Household characteristics ----------------
        house_type = np.random.choice(HOUSE_TYPES, p=[0.55, 0.35, 0.10])
        area_sqft = {
            "Apartment": np.random.normal(900, 200),
            "Independent House": np.random.normal(1500, 300),
            "Villa": np.random.normal(2800, 500),
        }[house_type]
        area_sqft = max(300, round(area_sqft))

        num_rooms = max(1, int(area_sqft // 350) + np.random.randint(0, 2))

        adults = np.random.randint(1, 4)
        children = np.random.randint(0, 3)
        seniors = np.random.choice([0, 1], p=[0.75, 0.25])
        family_members = adults + children + seniors

        work_from_home = np.random.choice([0, 1], p=[0.7, 0.3])

        # ---------------- Time / calendar features ----------------
        month = np.random.randint(1, 13)
        season = (
            "Summer" if month in [4, 5, 6] else
            "Winter" if month in [11, 12, 1] else
            "Spring" if month in [2, 3] else
            "Autumn"
        )
        day_of_week = np.random.choice(DAYS)
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        is_holiday = np.random.choice([0, 1], p=[0.92, 0.08])

        # ---------------- Weather ----------------
        base_temp = {"Summer": 38, "Winter": 14, "Spring": 26, "Autumn": 24}[season]
        outdoor_temp = round(np.random.normal(base_temp, 3), 1)
        humidity = round(np.clip(np.random.normal(55, 15), 10, 95), 1)

        # ---------------- Appliance usage (LOGIC-DRIVEN) ----------------
        # AC usage increases with temperature, family size, and WFH/weekend presence.
        at_home_factor = 1 + 0.15 * is_weekend + 0.2 * work_from_home
        temp_factor = max(0, (outdoor_temp - 20) / 10)  # more heat -> more AC
        ac_hours = np.clip(
            np.random.normal(temp_factor * 3 * at_home_factor, 1.5), 0, 16
        )

        fan_hours = np.clip(np.random.normal(6 + temp_factor * 2, 2), 0, 20)

        refrigerator_hours = 24  # always on, but compressor duty cycle handled via lower effective power

        # Family_Members now meaningfully drives TV/Laptop/Lighting/Washing usage,
        # reflecting that a bigger household runs more devices simultaneously
        # (multiple screens, more rooms lit, more laundry loads).
        family_factor = 1 + 0.12 * (family_members - 1)  # each extra member adds ~12% usage

        tv_hours = np.clip(
            np.random.normal((3 + is_weekend * 1.5 + work_from_home) * family_factor, 1.2), 0, 14
        )
        laptop_hours = np.clip(
            np.random.normal((3 + work_from_home * 3) * (1 + 0.08 * (family_members - 1)), 1.5), 0, 16
        )

        lighting_hours_per_room = np.clip(np.random.normal(6 * family_factor, 1.5), 2, 14)

        wm_prob = min(0.85, 0.35 + 0.10 * family_members)
        washing_machine_used = np.random.choice([0, 1], p=[1 - wm_prob, wm_prob])
        water_motor_hours = np.clip(np.random.normal(1.2, 0.5), 0, 4)
        microwave_hours = np.clip(np.random.normal(0.3, 0.15), 0, 1.5)
        geyser_hours = np.clip(
            np.random.normal(1.0 if season == "Winter" else 0.3, 0.4), 0, 3
        )
        ev_charging = np.random.choice([0, 1], p=[0.9, 0.1])
        solar_panel = np.random.choice([0, 1], p=[0.85, 0.15])
        battery_backup = np.random.choice([0, 1], p=[0.8, 0.2])

        electricity_rate = round(np.random.normal(22, 3), 2)  # currency units per kWh

        peak_hour_usage = np.random.choice([0, 1], p=[0.6, 0.4])  # used electricity during 6-10 PM peak

        # ---------------- ENERGY CALCULATION (Energy = Power x Time) ----------------
        energy = 0
        energy += ac_hours * POWER_RATING["AC"]
        energy += fan_hours * POWER_RATING["Fan"]
        energy += refrigerator_hours * POWER_RATING["Refrigerator"] * 0.35  # duty cycle ~35%
        energy += tv_hours * POWER_RATING["TV"]
        energy += laptop_hours * POWER_RATING["Laptop"]
        energy += lighting_hours_per_room * num_rooms * POWER_RATING["Lighting_Per_Room"]
        energy += washing_machine_used * POWER_RATING["Washing_Machine"]
        energy += water_motor_hours * POWER_RATING["Water_Motor"]
        energy += microwave_hours * POWER_RATING["Microwave"]
        energy += geyser_hours * POWER_RATING["Geyser"]
        energy += ev_charging * POWER_RATING["EV_Charging"]

        # Base/standby load (routers, chargers, clocks etc.)
        base_load = 0.5 + 0.1 * family_members
        energy += base_load

        # Solar panel offsets some consumption (self-generation)
        if solar_panel:
            energy *= np.random.uniform(0.65, 0.8)

        # Random real-world noise (+/- 5%)
        energy *= np.random.normal(1.0, 0.05)

        daily_consumption = round(max(1.0, energy), 2)

        rows.append({
            "House_ID": f"H{i:05d}",
            "House_Type": house_type,
            "Area_sqft": area_sqft,
            "Number_of_Rooms": num_rooms,
            "Family_Members": family_members,
            "Adults": adults,
            "Children": children,
            "Senior_Citizens": seniors,
            "Work_From_Home": work_from_home,
            "AC_Usage_Hours": round(ac_hours, 2),
            "Fan_Usage_Hours": round(fan_hours, 2),
            "Refrigerator_Hours": refrigerator_hours,
            "TV_Hours": round(tv_hours, 2),
            "Laptop_Hours": round(laptop_hours, 2),
            "Lighting_Hours_Per_Room": round(lighting_hours_per_room, 2),
            "Washing_Machine_Usage": washing_machine_used,
            "Water_Motor_Hours": round(water_motor_hours, 2),
            "Microwave_Usage_Hours": round(microwave_hours, 2),
            "Geyser_Usage_Hours": round(geyser_hours, 2),
            "EV_Charging": ev_charging,
            "Solar_Panel": solar_panel,
            "Battery_Backup": battery_backup,
            "Outdoor_Temperature": outdoor_temp,
            "Humidity": humidity,
            "Season": season,
            "Month": month,
            "Day_of_Week": day_of_week,
            "Is_Weekend": is_weekend,
            "Is_Holiday": is_holiday,
            "Peak_Hour_Usage": peak_hour_usage,
            "Electricity_Rate": electricity_rate,
            "Daily_Electricity_Consumption_kWh": daily_consumption,
        })

    df = pd.DataFrame(rows)

    # ---------------- Inject a small % of realistic messiness ----------------
    # Real-world data always has some missing values / duplicates.
    # We deliberately inject a small amount so our preprocessing step (Step 3)
    # has something genuine to clean -- this is intentional, not a bug.
    missing_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    for idx in missing_idx:
        col = np.random.choice(["Humidity", "Outdoor_Temperature", "Electricity_Rate"])
        df.loc[idx, col] = np.nan

    duplicate_rows = df.sample(n=15, random_state=1)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "electricity_consumption_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to: {out_path}")
    print("\nFirst 5 rows:\n", df.head())
    print("\nTarget variable stats:\n", df["Daily_Electricity_Consumption_kWh"].describe())
