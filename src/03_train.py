"""
Model training, time-split validation, and artifact persistence.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
FE_PATH = Path(__file__).resolve().parent / "02_feature_engineering.py"
_spec = importlib.util.spec_from_file_location("feature_engineering", FE_PATH)
_fe_mod = importlib.util.module_from_spec(_spec)
sys.modules["feature_engineering"] = _fe_mod
_spec.loader.exec_module(_fe_mod)
FeaturePipeline = _fe_mod.FeaturePipeline


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred_clipped = np.maximum(y_pred, 1.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    mape = float(mean_absolute_percentage_error(y_true, y_pred_clipped) * 100.0)
    r2 = float(r2_score(y_true, y_pred_clipped))

    return {
        "RMSE ($)": round(rmse, 2),
        "MAE ($)": round(mae, 2),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
    }


def print_metrics(model_name: str, metrics: Dict[str, float]) -> None:
    print(f"\n--- {model_name} Performance on October Holdout ---")
    for metric, val in metrics.items():
        print(f"  {metric:<10}: {val}")


def main() -> None:
    data_path = BASE_DIR / "data" / "train_test.csv"
    if not data_path.exists():
        print(f"File not found: {data_path}")
        raise SystemExit(1)

    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # Temporal split: Jan-Sep vs Oct
    split_date = "2025-10-01"
    train_df = df[df["date"] < split_date].copy()
    holdout_df = df[df["date"] >= split_date].copy()

    val_pipeline = FeaturePipeline()
    val_pipeline.fit(train_df)

    x_train, y_train = val_pipeline.transform(train_df, is_train=True)
    x_holdout, y_holdout = val_pipeline.transform(holdout_df, is_train=False)

    # 1. Ridge Model
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_model.fit(x_train, y_train)
    y_pred_ridge = ridge_model.predict(x_holdout)
    print_metrics("Ridge Baseline", evaluate_predictions(y_holdout.to_numpy(), y_pred_ridge))

    # 2. LightGBM Model
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 45,
        "min_child_samples": 15,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(x_train, y_train)
    y_pred_lgbm = lgbm_model.predict(x_holdout)
    print_metrics("LightGBM Regressor", evaluate_predictions(y_holdout.to_numpy(), y_pred_lgbm))

    # 3. Ensemble (85% LightGBM + 15% Ridge)
    y_pred_ensemble = 0.85 * y_pred_lgbm + 0.15 * y_pred_ridge
    ensemble_metrics = evaluate_predictions(y_holdout.to_numpy(), y_pred_ensemble)
    print_metrics("Ensemble (85% LGBM + 15% Ridge)", ensemble_metrics)

    # 4. Train final models on full dataset (Jan-Oct 48,000 rows)
    print("\nTraining final ensemble on full dataset (48,000 rows)...")
    full_pipeline = FeaturePipeline()
    full_pipeline.fit(df)
    x_full, y_full = full_pipeline.transform(df, is_train=True)

    final_lgbm = lgb.LGBMRegressor(**lgbm_params)
    final_lgbm.fit(x_full, y_full)

    final_ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    final_ridge.fit(x_full, y_full)

    # Save bundle
    output_dir = BASE_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "model.pkl"

    artifact = {
        "pipeline": full_pipeline,
        "lgbm_model": final_lgbm,
        "ridge_model": final_ridge,
        "validation_metrics": ensemble_metrics,
        "feature_importance": pd.DataFrame({
            "feature": full_pipeline.feature_columns,
            "importance": final_lgbm.feature_importances_,
        }).sort_values("importance", ascending=False).to_dict(orient="records"),
    }
    joblib.dump(artifact, artifact_path)
    print(f"\nTrained model artifact successfully saved to: {artifact_path}")


if __name__ == "__main__":
    main()
