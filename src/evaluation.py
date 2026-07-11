"""
evaluation.py
-------------
Generates diagnostic visualizations for the final tuned model:
  1. Actual vs Predicted scatter -- ideally points fall on the y=x diagonal
  2. Residual plot -- residuals should be randomly scattered around 0
     (a pattern here would mean the model is systematically wrong somewhere)
  3. Residual distribution -- should look roughly normal (bell-shaped)
  4. Feature importance (from the final model)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_features.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "eda")
TARGET = "Daily_Electricity_Consumption_kWh"
ID_COLS = ["House_ID"]
DROP_COLS = ["Refrigerator_Hours"]

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
X = df[feature_cols]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)
residuals = y_test.values - preds

# ---------------- 1. Actual vs Predicted ----------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.4, color="#2E86AB", s=20)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, "r--", label="Perfect Prediction (y=x)")
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title(f"Actual vs Predicted (R2 = {r2_score(y_test, preds):.4f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_actual_vs_predicted.png")
plt.close()

# ---------------- 2. Residual plot ----------------
plt.figure(figsize=(9, 5))
plt.scatter(preds, residuals, alpha=0.4, color="#F18F01", s=20)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Consumption (kWh)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_residual_plot.png")
plt.close()

# ---------------- 3. Residual distribution ----------------
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color="#6A4C93")
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual Distribution (should be ~normal, centered at 0)")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_residual_distribution.png")
plt.close()

# ---------------- 4. Feature importance ----------------
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 8))
    sns.barplot(x=importance.values, y=importance.index, palette="mako")
    plt.title("Top 15 Feature Importances (Final Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/12_feature_importance.png")
    plt.close()
    print("Top 15 features:\n", importance)

# ---------------- Adjusted R2 ----------------
n = len(y_test)
p = X_test.shape[1]
r2 = r2_score(y_test, preds)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"\nFinal Test Set Metrics:")
print(f"  MAE:  {mean_absolute_error(y_test, preds):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
print(f"  R2:   {r2:.4f}")
print(f"  Adjusted R2: {adj_r2:.4f}")
print(f"  Residual mean: {residuals.mean():.4f} (should be close to 0)")
print(f"  Residual std:  {residuals.std():.4f}")
print(f"\nAll evaluation plots saved to {OUT_DIR}")
