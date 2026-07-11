"""
feature_selection.py
----------------------
Evaluates feature importance using FOUR independent techniques, then combines
them into a single ranked shortlist. Using multiple techniques matters because
each one has blind spots:
  - Correlation:            only catches LINEAR relationships
  - Mutual Information:     catches linear AND non-linear relationships
  - Random Forest Importance: biased toward high-cardinality features
  - Permutation Importance: most reliable (measures actual performance drop),
                             but slower to compute
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_features.csv")
TARGET = "Daily_Electricity_Consumption_kWh"
ID_COLS = ["House_ID"]

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- 1. Correlation ----------------
corr_scores = X.corrwith(y).abs().sort_values(ascending=False)

# ---------------- 2. Mutual Information ----------------
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

# ---------------- 3. Random Forest Importance ----------------
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ---------------- 4. Permutation Importance ----------------
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

# ---------------- Combine into a ranked summary ----------------
summary = pd.DataFrame({
    "Correlation_Rank": corr_scores.rank(ascending=False),
    "MutualInfo_Rank": mi_series.rank(ascending=False),
    "RF_Importance_Rank": rf_importance.rank(ascending=False),
    "Permutation_Rank": perm_importance.rank(ascending=False),
})
summary["Avg_Rank"] = summary.mean(axis=1)
summary = summary.sort_values("Avg_Rank")

print("=" * 70)
print("TOP 15 FEATURES (combined ranking across 4 methods)")
print("=" * 70)
print(summary.head(15).round(1))

print("\nTop 10 by Random Forest Importance (raw values):")
print(rf_importance.head(10).round(4))

print("\nBottom 10 (weakest features -- candidates for removal):")
print(summary.tail(10).round(1))

summary.to_csv(os.path.join(os.path.dirname(__file__), "..", "reports", "feature_importance_ranking.csv"))
print("\nSaved full ranking to reports/feature_importance_ranking.csv")
