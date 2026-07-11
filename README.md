# ⚡ Smart Electricity Consumption Prediction & Energy Optimization System

A production-style Machine Learning project that predicts household daily electricity
consumption and generates personalized energy-saving recommendations.

## Live Links
https://qt2rt5qk-8501.inc1.devtunnels.ms/      

https://smartelectricityconsumptionprediction.streamlit.app/
## Project Overview

This system covers the complete ML lifecycle:
Original Dataset Creation → Preprocessing → EDA → Feature Engineering →
Model Training & Comparison → Evaluation → Prediction System →
Recommendation Engine → Reporting.

**Best Model:** Gradient Boosting Regressor (tuned) — **R² = 0.9726**, MAE = 0.58 kWh, RMSE = 0.76 kWh

## Folder Structure

```
electricity_prediction/
│
├── data/
│   ├── raw/                       # original synthetic dataset
│   └── processed/                 # cleaned, encoded, scaled, feature-engineered data
│
├── notebooks/                     # exploration notebooks (optional)
│
├── src/
│   ├── generate_dataset.py        # Step 1-2: synthetic dataset generator
│   ├── preprocessing.py           # Step 3: cleaning, encoding, scaling
│   ├── visualization.py           # Step 4: EDA plots
│   ├── feature_engineering.py     # Step 5: new features
│   ├── feature_selection.py       # Step 5: 4-method feature importance
│   ├── train.py                   # Step 6: model training + tuning
│   ├── evaluation.py              # Step 7: diagnostic plots + metrics
│   ├── recommendation_engine.py   # Step 8: energy-saving logic
│   ├── predict.py                 # unified prediction pipeline
│   └── report_generator.py        # Step 10: Excel/CSV report generation
│
├── models/                        # trained model, scaler, feature column order
├── reports/                       # EDA plots, model comparison, generated reports
├── dashboard/                     # (Streamlit app lives at project root: app.py)
├── screenshots/                   # dashboard screenshots for submission
│
├── app.py                         # Streamlit dashboard (Step 9)
├── requirements.txt
└── README.md
```

## Dataset

**100% original, synthetically generated** (no Kaggle/UCI/GitHub data used), with
**2000 records and 30 raw features** before engineering (53 after encoding + feature
engineering). Generation is physics-informed: `Energy (kWh) = Power (kW) × Time (hours)`
per appliance, with logical dependencies (e.g. Season → Temperature → AC Hours →
Consumption) and controlled random noise. See `src/generate_dataset.py` for full logic
and inline documentation of every design decision.

## How to Run

```bash
pip install -r requirements.txt

# Regenerate the full pipeline from scratch (optional -- outputs already included)
python src/generate_dataset.py
python src/preprocessing.py
python src/visualization.py
python src/feature_engineering.py
python src/feature_selection.py
python src/train.py
python src/evaluation.py
python src/report_generator.py

# Launch the dashboard
streamlit run app.py
```

## Model Comparison

| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.58** | **0.76** | **0.9726** | 0.9717 |
| XGBoost | 0.63 | 0.84 | 0.9663 | 0.9610 |
| Linear Regression | 0.65 | 0.87 | 0.9637 | 0.9637 |
| Extra Trees | 0.65 | 0.88 | 0.9632 | 0.9621 |
| Random Forest | 0.65 | 0.88 | 0.9626 | 0.9596 |
| Decision Tree | 0.90 | 1.22 | 0.9282 | 0.9260 |
| SVR (unscaled) | 3.09 | 3.98 | 0.2401 | 0.2302 |

**Note on SVR:** SVR was intentionally run on unscaled features in this comparison to
keep the pipeline uniform across model types. Its poor performance illustrates a real
ML principle: distance-based/kernel algorithms (SVR, KNN) are highly scale-sensitive and
require `StandardScaler`-transformed input to perform correctly — unlike tree-based
models, which are scale-invariant. A scaled version (`data/processed/electricity_scaled.csv`)
is provided for this purpose.

## Key Business Insights (from EDA)

- AC usage and outdoor temperature are the strongest consumption drivers (r = 0.78, 0.59)
- Households with solar panels show ~27% lower average consumption (8.89 vs 12.17 kWh/day)
- EV charging households average 17.58 kWh/day vs 11.03 kWh/day without — the single
  largest reducible load
- Summer consumption is highest across all seasons; Winter is lowest
- `Family_Members` alone is a weak predictor (r ≈ 0.05) — actual appliance *behavior*
  matters far more than headcount, a realistic and non-obvious finding

## Feature Engineering Highlights

The engineered `Power_Usage_Index` (a physics-weighted sum of appliance hours) achieved
a stronger correlation with the target (r = 0.81) than any single raw feature, and
accounted for ~65% of the final model's feature importance — demonstrating the value of
domain-informed feature construction over relying on raw inputs alone.

## Explainability

Feature importance was validated using four independent methods (correlation, mutual
information, Random Forest importance, permutation importance) to avoid relying on any
single method's blind spots. Results are in `reports/feature_importance_ranking.csv`.

## Limitations & Future Improvements

- Dataset is synthetic; real smart-meter data would capture consumption patterns
  (e.g. appliance-specific anomalies) that formula-based generation cannot fully replicate
- Recommendation engine is rule-based; a learned/RL-based recommender could personalize
  further using historical household behavior
- SHAP-based per-prediction explanations could be added to the dashboard for deeper
  transparency (feature importance is currently global, not per-prediction)
- Weather API integration would allow real-time rather than user-entered temperature/humidity

## Author's Note

This project was built as a guided learning exercise, with every design decision
(power ratings, outlier handling, encoding choice, model selection, etc.) explained
and justified rather than applied as a black box.
