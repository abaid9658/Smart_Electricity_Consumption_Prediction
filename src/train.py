"""
train.py
--------
Trains and compares 6 regression algorithms:
  1. Linear Regression      - baseline, interpretable, assumes linear relationships
  2. Decision Tree          - captures non-linearity, prone to overfitting alone
  3. Random Forest          - ensemble of trees, reduces overfitting via averaging
  4. Gradient Boosting      - sequential error-correction, usually strong on tabular data
  5. Extra Trees            - like RF but more randomized splits, often faster/robust
  6. XGBoost                - optimized gradient boosting, industry standard for tabular ML

We use:
  - Train/Test split (80/20) for a final holdout evaluation
  - 5-Fold Cross Validation for robust performance estimates (not just one split's luck)
  - RandomizedSearchCV for hyperparameter tuning on the top candidates (faster than
    exhaustive GridSearchCV while still exploring a wide parameter space)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "electricity_features.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TARGET = "Daily_Electricity_Consumption_kWh"
ID_COLS = ["House_ID"]
DROP_COLS = ["Refrigerator_Hours"]  # zero variance -- no information

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_all_models():
    X_train, X_test, y_train, y_test, feature_cols = load_and_split()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "SVR": SVR(kernel="rbf", C=10),
    }

    results = {}
    trained_models = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 70)
    print("TRAINING & COMPARING MODELS (5-Fold Cross Validation + Holdout Test)")
    print("=" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        metrics["CV_R2_Mean"] = cv_scores.mean()
        metrics["CV_R2_Std"] = cv_scores.std()

        results[name] = metrics
        trained_models[name] = model

        print(f"\n{name}:")
        print(f"  MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}  "
              f"MAPE={metrics['MAPE']:.2f}%  CV_R2={metrics['CV_R2_Mean']:.4f}(+/-{metrics['CV_R2_Std']:.4f})")

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (sorted by R2, best first)")
    print("=" * 70)
    print(results_df.round(4))

    best_model_name = results_df.index[0]
    print(f"\n>>> BEST MODEL (pre-tuning): {best_model_name}")

    return trained_models, results_df, X_train, X_test, y_train, y_test, feature_cols, best_model_name


def tune_best_model(best_model_name, X_train, y_train):
    """Hyperparameter tuning via RandomizedSearchCV on the best-performing model family."""
    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING: {best_model_name}")
    print("=" * 70)

    param_grids = {
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.7, 0.8, 1.0],
            },
        ),
        "XGBoost": (
            xgb.XGBRegressor(random_state=42, verbosity=0),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.8, 1.0],
            },
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 5, 10]},
        ),
        "Linear Regression": (LinearRegression(), {}),
        "SVR": (
            SVR(),
            {"C": [1, 10, 50, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        ),
    }

    model, grid = param_grids[best_model_name]

    if not grid:  # Linear Regression has no hyperparameters to tune
        model.fit(X_train, y_train)
        return model

    search = RandomizedSearchCV(
        model, grid, n_iter=20, cv=5, scoring="r2", random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV R2 during search: {search.best_score_:.4f}")
    return search.best_estimator_


if __name__ == "__main__":
    trained_models, results_df, X_train, X_test, y_train, y_test, feature_cols, best_model_name = train_all_models()

    tuned_model = tune_best_model(best_model_name, X_train, y_train)
    tuned_metrics = evaluate(tuned_model, X_test, y_test)

    print("\n" + "=" * 70)
    print("FINAL TUNED MODEL PERFORMANCE (on held-out test set)")
    print("=" * 70)
    for k, v in tuned_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save the best/tuned model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(tuned_model, model_path)
    print(f"\nBest model saved to: {model_path}")

    # Save feature column order (CRITICAL for the prediction app to build inputs correctly)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    # Save comparison results
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    with open(os.path.join(REPORTS_DIR, "best_model_info.json"), "w") as f:
        json.dump({"best_model_name": best_model_name, "tuned_metrics": tuned_metrics}, f, indent=2)

    print(f"Model comparison saved to: {REPORTS_DIR}/model_comparison.csv")
    print(f"\n>>> WINNER: {best_model_name} (tuned) <<<")
