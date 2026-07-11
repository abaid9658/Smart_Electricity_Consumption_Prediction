"""
report_generator.py
---------------------
Generates the final project reports: Dataset Summary, Model Performance
Report, and an Excel workbook combining everything -- exported to reports/.
"""

import pandas as pd
import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "electricity_features.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TARGET = "Daily_Electricity_Consumption_kWh"

df = pd.read_csv(DATA_PATH)

# ---------------- Dataset Summary ----------------
dataset_summary = pd.DataFrame({
    "Metric": ["Total Records", "Total Features", "Missing Values", "Duplicate Rows",
               "Target Mean (kWh)", "Target Std (kWh)", "Target Min (kWh)", "Target Max (kWh)"],
    "Value": [len(df), df.shape[1] - 2, df.isnull().sum().sum(), df.duplicated().sum(),
              round(df[TARGET].mean(), 2), round(df[TARGET].std(), 2),
              round(df[TARGET].min(), 2), round(df[TARGET].max(), 2)],
})

# ---------------- Model Performance ----------------
model_comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
model_comparison = pd.read_csv(model_comparison_path, index_col=0) if os.path.exists(model_comparison_path) else pd.DataFrame()

best_model_info_path = os.path.join(REPORTS_DIR, "best_model_info.json")
if os.path.exists(best_model_info_path):
    with open(best_model_info_path) as f:
        best_model_info = json.load(f)
else:
    best_model_info = {}

# ---------------- Energy Consumption Report ----------------
energy_report = pd.DataFrame({
    "Segment": ["Overall", "Solar Panel Households", "No Solar Households",
                "Weekday", "Weekend", "EV Charging", "No EV Charging"],
    "Avg_Daily_kWh": [
        df[TARGET].mean(),
        df[df.Solar_Panel == 1][TARGET].mean(),
        df[df.Solar_Panel == 0][TARGET].mean(),
        df[df.Is_Weekend == 0][TARGET].mean(),
        df[df.Is_Weekend == 1][TARGET].mean(),
        df[df.EV_Charging == 1][TARGET].mean(),
        df[df.EV_Charging == 0][TARGET].mean(),
    ],
}).round(2)

# ---------------- Monthly Forecast (sample projection) ----------------
avg_daily = df[TARGET].mean()
monthly_forecast = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "Est_Avg_Daily_kWh": [avg_daily * f for f in
                           [0.85, 0.88, 0.95, 1.15, 1.30, 1.35, 1.25, 1.20, 1.05, 0.95, 0.88, 0.85]],
})
monthly_forecast["Est_Monthly_kWh"] = (monthly_forecast["Est_Avg_Daily_kWh"] * 30).round(2)
monthly_forecast["Est_Avg_Daily_kWh"] = monthly_forecast["Est_Avg_Daily_kWh"].round(2)

# ---------------- Write everything to Excel (multi-sheet) ----------------
excel_path = os.path.join(REPORTS_DIR, "Project_Report.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    dataset_summary.to_excel(writer, sheet_name="Dataset Summary", index=False)
    if not model_comparison.empty:
        model_comparison.to_excel(writer, sheet_name="Model Comparison")
    energy_report.to_excel(writer, sheet_name="Energy Consumption Report", index=False)
    monthly_forecast.to_excel(writer, sheet_name="Monthly Forecast", index=False)

# Also save individual CSVs
dataset_summary.to_csv(os.path.join(REPORTS_DIR, "dataset_summary.csv"), index=False)
energy_report.to_csv(os.path.join(REPORTS_DIR, "energy_consumption_report.csv"), index=False)
monthly_forecast.to_csv(os.path.join(REPORTS_DIR, "monthly_forecast.csv"), index=False)

print("Reports generated:")
print(f"  - {excel_path}")
print(f"  - {REPORTS_DIR}/dataset_summary.csv")
print(f"  - {REPORTS_DIR}/energy_consumption_report.csv")
print(f"  - {REPORTS_DIR}/monthly_forecast.csv")
print("\nDataset Summary:\n", dataset_summary.to_string(index=False))
print("\nEnergy Consumption Report:\n", energy_report.to_string(index=False))
print("\nBest Model Info:", best_model_info)
