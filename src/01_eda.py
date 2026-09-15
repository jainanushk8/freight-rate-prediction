from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_test.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    return df


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def basic_info(df: pd.DataFrame) -> None:
    section("Basic Info")
    print(f"Shape            : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range       : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique load IDs  : {df['load_id'].nunique():,}")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")


def missing_values(df: pd.DataFrame) -> None:
    section("Missing Values")
    nulls = df.isnull().sum()
    pct = (nulls / len(df) * 100).round(2)
    report = pd.DataFrame({"null_count": nulls, "null_pct": pct})
    report = report[report["null_count"] > 0]
    if report.empty:
        print("No missing values found.")
    else:
        print(report.to_string())


def target_distribution(df: pd.DataFrame) -> None:
    section("Target: posted_rate")
    stats = df["posted_rate"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(stats.to_string())
    skew = df["posted_rate"].skew()
    kurt = df["posted_rate"].kurt()
    print(f"\nSkewness : {skew:.4f}")
    print(f"Kurtosis : {kurt:.4f}")
    mean, std = df["posted_rate"].mean(), df["posted_rate"].std()
    n_outliers = ((df["posted_rate"] - mean).abs() > 3 * std).sum()
    print(f"Outliers (>3 std): {n_outliers} rows ({n_outliers / len(df) * 100:.2f}%)")


def rate_by_equipment(df: pd.DataFrame) -> None:
    section("posted_rate by Equipment Type")
    grp = (
        df.groupby("equipment")["posted_rate"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    print(grp.to_string())


def rate_by_month(df: pd.DataFrame) -> None:
    section("posted_rate by Month")
    df = df.copy()
    df["month"] = df["date"].dt.month
    grp = (
        df.groupby("month")["posted_rate"]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct"]
    grp.index = month_names[: len(grp)]
    print(grp.to_string())


def rate_by_weekday(df: pd.DataFrame) -> None:
    section("posted_rate by Day of Week")
    df = df.copy()
    df["weekday"] = df["date"].dt.day_name()
    df["dow"] = df["date"].dt.dayofweek
    grp = (
        df.groupby(["dow", "weekday"])["posted_rate"]
        .agg(["count", "mean", "median"])
        .round(2)
        .reset_index()
        .set_index("weekday")
        .drop(columns=["dow"])
    )
    print(grp.to_string())


def feature_correlations(df: pd.DataFrame) -> None:
    section("Correlations with posted_rate")
    numeric_cols = ["distance", "weight", "market_index", "quote_signal"]
    corr = df[numeric_cols + ["posted_rate"]].corr()["posted_rate"].drop("posted_rate")
    print(corr.round(4).to_string())


def city_pairs(df: pd.DataFrame) -> None:
    section("Top 15 City Pairs by Volume")
    df = df.copy()
    df["route"] = df["pickup"] + " -> " + df["delivery"]
    grp = (
        df.groupby("route")["posted_rate"]
        .agg(["count", "mean", "median"])
        .sort_values("count", ascending=False)
        .head(15)
        .round(2)
    )
    print(grp.to_string())


def distance_stats(df: pd.DataFrame) -> None:
    section("Distance Distribution")
    stats = df["distance"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
    print(stats.to_string())
    print(f"\nRate per mile (mean) : ${(df['posted_rate'] / df['distance']).mean():.4f}")


def market_signals(df: pd.DataFrame) -> None:
    section("Market Signals")
    idx = df["market_index"].dropna()
    print(f"market_index — present: {len(idx):,}, missing: {df['market_index'].isna().sum()}")
    print(idx.describe().round(4).to_string())
    print(f"\nquote_signal — present: {df['quote_signal'].notna().sum():,}, missing: {df['quote_signal'].isna().sum()}")
    print(df["quote_signal"].describe().round(4).to_string())


def time_split_preview(df: pd.DataFrame) -> None:
    section("Proposed Time-Based Train / Holdout Split")
    train = df[df["date"] < "2025-10-01"]
    holdout = df[df["date"] >= "2025-10-01"]
    print(f"Train    (Jan-Sep): {len(train):,} rows  | rate mean: ${train['posted_rate'].mean():.2f}")
    print(f"Holdout  (Oct)    : {len(holdout):,} rows  | rate mean: ${holdout['posted_rate'].mean():.2f}")
    print(f"\nValidation set spans: Nov-Dec 2025  (12,000 rows - no target)")


def unique_cities(df: pd.DataFrame) -> None:
    section("Unique Cities")
    all_cities = set(df["pickup"].unique()) | set(df["delivery"].unique())
    print(f"Unique pickup cities   : {df['pickup'].nunique()}")
    print(f"Unique delivery cities : {df['delivery'].nunique()}")
    print(f"Total unique cities    : {len(all_cities)}")
    print(f"\nPickup cities:\n{sorted(df['pickup'].unique())}")


def main() -> None:
    print("Loading training data...")
    df = load_data()

    basic_info(df)
    missing_values(df)
    target_distribution(df)
    rate_by_equipment(df)
    rate_by_month(df)
    rate_by_weekday(df)
    feature_correlations(df)
    city_pairs(df)
    distance_stats(df)
    market_signals(df)
    time_split_preview(df)
    unique_cities(df)

    print("\n\nEDA complete.")


if __name__ == "__main__":
    main()
