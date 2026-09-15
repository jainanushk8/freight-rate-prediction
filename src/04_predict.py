"""
Generates final submission predictions:
1. validation_predictions.csv (12,000 loads)
2. data/december_chart_inputs.csv (31 days, exact 7 columns)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FE_PATH = Path(__file__).resolve().parent / "02_feature_engineering.py"
_spec = importlib.util.spec_from_file_location("feature_engineering", FE_PATH)
_fe_mod = importlib.util.module_from_spec(_spec)
sys.modules["feature_engineering"] = _fe_mod
_spec.loader.exec_module(_fe_mod)
FeaturePipeline = _fe_mod.FeaturePipeline

REQUIRED_DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def main() -> None:
    model_path = BASE_DIR / "models" / "model.pkl"
    if not model_path.exists():
        print(f"Model artifact not found at {model_path}. Please run src/03_train.py first.")
        raise SystemExit(1)

    print(f"Loading model artifact from: {model_path}")
    artifact = joblib.load(model_path)
    pipeline: FeaturePipeline = artifact["pipeline"]
    lgbm_model = artifact["lgbm_model"]
    ridge_model = artifact["ridge_model"]

    # 1. Predict validation.csv
    val_path = BASE_DIR / "data" / "validation.csv"
    print(f"\nReading validation data: {val_path}")
    val_df = pd.read_csv(val_path)

    x_val, _ = pipeline.transform(val_df, is_train=False)
    pred_lgbm = lgbm_model.predict(x_val)
    pred_ridge = ridge_model.predict(x_val)
    val_preds = np.round(np.maximum(0.85 * pred_lgbm + 0.15 * pred_ridge, 10.0), 2)

    template_path = BASE_DIR / "data" / "validation_predictions_template.csv"
    if template_path.exists():
        sub_df = pd.read_csv(template_path)
        sub_df["predicted_rate"] = val_preds
    else:
        sub_df = pd.DataFrame({
            "load_id": val_df["load_id"],
            "predicted_rate": val_preds,
        })

    out_val_path = BASE_DIR / "validation_predictions.csv"
    sub_df.to_csv(out_val_path, index=False)
    print(f"Saved {len(sub_df):,} validation predictions to: {out_val_path}")
    print(f"Validation rate summary:\n{sub_df['predicted_rate'].describe().round(2)}")

    # 2. Predict december_chart_inputs.csv
    dec_path = BASE_DIR / "data" / "december_chart_inputs.csv"
    print(f"\nReading December inputs: {dec_path}")
    dec_df = pd.read_csv(dec_path)

    x_dec, _ = pipeline.transform(dec_df, is_train=False)
    dec_lgbm = lgbm_model.predict(x_dec)
    dec_ridge = ridge_model.predict(x_dec)
    dec_preds = np.round(np.maximum(0.85 * dec_lgbm + 0.15 * dec_ridge, 10.0), 2)

    dec_df["predicted_rate"] = dec_preds

    # Ensure strictly the exact 7 columns in the required order
    dec_output = dec_df[REQUIRED_DECEMBER_COLUMNS].copy()
    dec_output.to_csv(dec_path, index=False)
    print(f"\nFilled {len(dec_output)} December predictions in: {dec_path}")
    print(f"Columns written: {list(dec_output.columns)}")
    print(f"December rate summary:\n{dec_output['predicted_rate'].describe().round(2)}")


if __name__ == "__main__":
    main()
