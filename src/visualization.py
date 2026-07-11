"""
eda.py
------
Generates publication-quality EDA visualizations from the CLEANED (unscaled)
dataset. We use the unscaled version for EDA because scaled numbers (mean=0,
std=1) are unreadable to humans -- e.g. "AC_Usage_Hours = -0.42" means
nothing intuitively, but "AC_Usage_Hours = 3.2 hours" does.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_cleaned.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "eda")
TARGET = "Daily_Electricity_Consumption_kWh"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# We also want the original (pre-encoding) categorical columns for readable plots.
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "electricity_consumption_raw.csv")
df_raw = pd.read_csv(RAW_PATH).drop_duplicates(subset=[c for c in pd.read_csv(RAW_PATH).columns if c != "House_ID"])


# ---------------------------------------------------------------
# 1. Target Variable Distribution
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df[TARGET], kde=True, ax=axes[0], color="#2E86AB")
axes[0].set_title("Distribution of Daily Electricity Consumption (kWh)")
axes[0].set_xlabel("kWh")
sns.boxplot(x=df[TARGET], ax=axes[1], color="#F18F01")
axes[1].set_title("Boxplot: Daily Electricity Consumption")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_target_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 2. Correlation Heatmap
# ---------------------------------------------------------------
plt.figure(figsize=(16, 12))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Correlation Heatmap - All Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png")
plt.close()

# Also save the top correlations with target as a readable bar chart
target_corr = corr[TARGET].drop(TARGET).sort_values()
plt.figure(figsize=(8, 10))
colors = ["#C0392B" if v < 0 else "#2E86AB" for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors)
plt.title("Feature Correlation with Daily Electricity Consumption")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_target_correlation_bar.png")
plt.close()

# ---------------------------------------------------------------
# 3. Temperature vs Consumption (scatter, colored by AC usage)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["Outdoor_Temperature"], df[TARGET], c=df["AC_Usage_Hours"],
                  cmap="autumn_r", alpha=0.6, s=20)
plt.colorbar(sc, label="AC Usage Hours")
plt.xlabel("Outdoor Temperature (°C)")
plt.ylabel("Daily Consumption (kWh)")
plt.title("Temperature vs Consumption (colored by AC usage)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_temperature_vs_consumption.png")
plt.close()

# ---------------------------------------------------------------
# 4. Family Size vs Consumption
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="Family_Members", y=TARGET, data=df, palette="viridis")
plt.title("Family Size vs Daily Electricity Consumption")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_family_size_vs_consumption.png")
plt.close()

# ---------------------------------------------------------------
# 5. Room Count vs Consumption
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Number_of_Rooms", y=TARGET, hue="House_Type" if "House_Type" in df_raw.columns else None,
                 data=df_raw, alpha=0.5)
plt.title("Room Count vs Consumption (by House Type)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_rooms_vs_consumption.png")
plt.close()

# ---------------------------------------------------------------
# 6. Seasonal Electricity Usage
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
season_order = ["Winter", "Spring", "Summer", "Autumn"]
sns.boxplot(x="Season", y=TARGET, data=df_raw, order=season_order, palette="coolwarm")
plt.title("Seasonal Electricity Usage")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_seasonal_usage.png")
plt.close()

# ---------------------------------------------------------------
# 7. Weekday vs Weekend + Solar Panel impact
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(x="Is_Weekend", y=TARGET, data=df, ax=axes[0], palette="Set2")
axes[0].set_xticklabels(["Weekday", "Weekend"])
axes[0].set_title("Weekday vs Weekend Consumption")

sns.boxplot(x="Solar_Panel", y=TARGET, data=df, ax=axes[1], palette="Set3")
axes[1].set_xticklabels(["No Solar", "Has Solar"])
axes[1].set_title("Solar Panel Impact on Consumption")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_weekend_solar_impact.png")
plt.close()

# ---------------------------------------------------------------
# 8. Missing value analysis (post-cleaning proof)
# ---------------------------------------------------------------
missing = df.isnull().sum()
print("Missing values after cleaning:", missing.sum())

# ---------------------------------------------------------------
# 9. Key numeric summary
# ---------------------------------------------------------------
print("\n=== Business Insight Numbers ===")
print(f"Avg consumption (No Solar): {df[df.Solar_Panel==0][TARGET].mean():.2f} kWh")
print(f"Avg consumption (Has Solar): {df[df.Solar_Panel==1][TARGET].mean():.2f} kWh")
print(f"Avg consumption (Weekday): {df[df.Is_Weekend==0][TARGET].mean():.2f} kWh")
print(f"Avg consumption (Weekend): {df[df.Is_Weekend==1][TARGET].mean():.2f} kWh")
print(f"Avg consumption (EV Charging=1): {df[df.EV_Charging==1][TARGET].mean():.2f} kWh")
print(f"Avg consumption (EV Charging=0): {df[df.EV_Charging==0][TARGET].mean():.2f} kWh")
print("\nTop 5 correlated features with target:")
print(target_corr.abs().sort_values(ascending=False).head(5))

print("\nAll EDA plots saved to:", OUT_DIR)
