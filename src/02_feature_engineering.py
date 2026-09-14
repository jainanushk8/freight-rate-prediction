"""
02_feature_engineering.py
Feature extraction and transformation pipeline for freight rate prediction.
Includes city coordinate memory and cyclical temporal encodings.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

EQUIPMENT_MAP = {
    "Dry Van": 0,
    "Flatbed": 1,
    "Reefer": 2,
}


def compute_haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute great-circle distance in miles between two coordinate sets."""
    r_miles = 3958.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r_miles * c


class FeaturePipeline:
    """Fit-transform pipeline for preprocessing tabular load features."""

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing
        self.imputers: Dict[str, float] = {}
        self.target_encodings: Dict[str, Dict[str, float]] = {}
        self.city_coords: Dict[str, Tuple[float, float]] = {}
        self.dow_encoding: Dict[int, float] = {}
        self.global_target_mean: float = 0.0
        self.feature_columns: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: str = "posted_rate") -> "FeaturePipeline":
        """Compute imputation values, city coordinates, and encodings from training data."""
        self.imputers["weight"] = float(df["weight"].median())
        self.imputers["market_index"] = float(df["market_index"].mean())
        self.imputers["quote_signal"] = float(df["quote_signal"].mean())

        # Remember average coordinates for all cities
        for _, row in df.iterrows():
            p_city = row["pickup"]
            d_city = row["delivery"]
            if p_city not in self.city_coords and pd.notna(row.get("pickup_lat")):
                self.city_coords[p_city] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
            if d_city not in self.city_coords and pd.notna(row.get("delivery_lat")):
                self.city_coords[d_city] = (float(row["delivery_lat"]), float(row["delivery_lon"]))

        if target_col in df.columns:
            self.global_target_mean = float(df[target_col].mean())
            for col in ["pickup", "delivery"]:
                stats = df.groupby(col)[target_col].agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                smoothed = (counts * means + self.smoothing * self.global_target_mean) / (
                    counts + self.smoothing
                )
                self.target_encodings[col] = smoothed.to_dict()

            # Day of week seasonality encoding
            dow_series = pd.to_datetime(df["date"]).dt.dayofweek
            dow_stats = df.groupby(dow_series)[target_col].mean()
            self.dow_encoding = (dow_stats - self.global_target_mean).to_dict()

        return self

    def transform(
        self,
        df: pd.DataFrame,
        is_train: bool = False,
        target_col: str = "posted_rate",
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform raw input DataFrame into engineered feature matrix."""
        data = df.copy()

        # Date transformations
        date_series = pd.to_datetime(data["date"])
        data["month"] = date_series.dt.month
        data["day_of_month"] = date_series.dt.day
        data["day_of_week"] = date_series.dt.dayofweek
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(float)
        data["is_month_end"] = (data["day_of_month"] >= 26).astype(float)

        # Cyclical month encodings (smooth annual cycle connecting Dec to Jan)
        data["sin_month"] = np.sin(2.0 * np.pi * data["month"] / 12.0)
        data["cos_month"] = np.cos(2.0 * np.pi * data["month"] / 12.0)
        data["dow_effect"] = data["day_of_week"].map(self.dow_encoding).fillna(0.0)

        # Imputation
        data["weight"] = data["weight"].fillna(self.imputers.get("weight", 32000.0))
        data["market_index"] = data.get("market_index", pd.Series(dtype=float))
        data["market_index"] = data["market_index"].fillna(self.imputers.get("market_index", 1.08))

        data["quote_signal"] = data.get("quote_signal", pd.Series(dtype=float))
        data["quote_signal"] = data["quote_signal"].fillna(self.imputers.get("quote_signal", 2.06))

        # Equipment encoding
        data["equipment_code"] = data["equipment"].map(EQUIPMENT_MAP).fillna(0).astype(int)

        # Target encoding for categorical routes
        for col in ["pickup", "delivery"]:
            mapping = self.target_encodings.get(col, {})
            encoded_col = f"{col}_target_enc"
            data[encoded_col] = data[col].map(mapping).fillna(self.global_target_mean)

        # Non-linear transformations
        data["log_distance"] = np.log1p(np.maximum(data["distance"], 0.0))
        data["log_weight"] = np.log1p(np.maximum(data["weight"], 0.0))

        # Spatial coordinates (fill from learned city coords if missing)
        if not {"pickup_lat", "pickup_lon"}.issubset(data.columns):
            p_coords = data["pickup"].map(self.city_coords)
            data["pickup_lat"] = p_coords.apply(lambda c: c[0] if isinstance(c, tuple) else 35.0)
            data["pickup_lon"] = p_coords.apply(lambda c: c[1] if isinstance(c, tuple) else -90.0)

        if not {"delivery_lat", "delivery_lon"}.issubset(data.columns):
            d_coords = data["delivery"].map(self.city_coords)
            data["delivery_lat"] = d_coords.apply(lambda c: c[0] if isinstance(c, tuple) else 35.0)
            data["delivery_lon"] = d_coords.apply(lambda c: c[1] if isinstance(c, tuple) else -90.0)

        data["haversine_dist"] = compute_haversine_distance(
            data["pickup_lat"].to_numpy(),
            data["pickup_lon"].to_numpy(),
            data["delivery_lat"].to_numpy(),
            data["delivery_lon"].to_numpy(),
        )
        data["route_circuitousness"] = data["distance"] / np.maximum(data["haversine_dist"], 1.0)

        # Quote and market interactions
        data["quote_x_dist"] = data["quote_signal"] * data["distance"]
        data["market_x_dist"] = data["market_index"] * data["distance"]
        data["rate_per_mile_signal"] = data["quote_signal"] / np.maximum(data["distance"], 1.0)
        data["weight_x_dist"] = data["weight"] * data["distance"]

        feature_cols = [
            "distance",
            "log_distance",
            "weight",
            "log_weight",
            "equipment_code",
            "market_index",
            "quote_signal",
            "pickup_target_enc",
            "delivery_target_enc",
            "day_of_month",
            "day_of_week",
            "is_weekend",
            "is_month_end",
            "sin_month",
            "cos_month",
            "dow_effect",
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "haversine_dist",
            "route_circuitousness",
            "quote_x_dist",
            "market_x_dist",
            "rate_per_mile_signal",
            "weight_x_dist",
        ]

        if is_train:
            self.feature_columns = feature_cols

        x_out = data[feature_cols].copy()
        y_out = data[target_col].copy() if target_col in data.columns else None

        return x_out, y_out

    def save(self, filepath: str) -> None:
        """Persist fitted pipeline configuration to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipeline":
        """Load fitted pipeline configuration from disk."""
        return joblib.load(filepath)
