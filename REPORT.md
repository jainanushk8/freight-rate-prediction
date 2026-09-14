
# Spotter ML Engineer Assessment: Technical Report
**Candidate Assessment Submission**

---

## Executive Summary

This report documents the end-to-end development of a freight rate prediction model for the Spotter Machine Learning Engineer assessment. The objective is to forecast spot freight rates (`posted_rate`) across continental US freight lanes for November and December 2025. 

Using a combination of non-linear gradient-boosted decision trees (LightGBM) and regularized linear regression (Ridge) with domain-specific spatial and interaction features, our final weighted ensemble achieved a **6.51% Mean Absolute Percentage Error (MAPE)** and an **R² of 0.8131** on an out-of-time monthly holdout set.

---

## 1. Key Findings from Exploratory Data Analysis

Exploration of `data/train_test.csv` (48,000 loads from January through October 2025) yielded key domain insights:

1. **Haul Distance is the Primary Driver**:
   - The Pearson correlation between `distance` and `posted_rate` is **0.9085**, explaining over 82% of target variance linearly.
   - The overall historical average rate per mile is **$2.22/mile**.
   - Short hauls (<300 miles) exhibit higher rates per mile ($3.00–$4.50/mile) due to fixed loading/unloading overhead, whereas long hauls (>1,500 miles) compress toward $1.90–$2.10/mile.

2. **Equipment Type Price Hierarchy**:
   - **Dry Van** (56.7% of volume): Baseline median rate of **$1,953.04** (Mean: $2,271.55).
   - **Flatbed** (18.2% of volume): Median rate of **$2,076.81** (Mean: $2,445.09; +7.6% premium over Dry Van).
   - **Reefer / Refrigerated** (25.1% of volume): Median rate of **$2,196.67** (Mean: $2,553.64; +12.4% premium over Dry Van).

3. **Temporal & Seasonal Patterns**:
   - Average rates peak during early summer (June mean: **$2,497.03**) and taper into late summer/early autumn (August mean: **$2,338.41**).
   - Day-of-week seasonality shows midweek peaks (Wednesday/Thursday average ~$2,404) and weekend dips (Sunday average ~$2,335), reflecting commercial shipper dispatch schedules.

4. **Target Distribution**:
   - The target `posted_rate` is right-skewed (skewness: 1.90, kurtosis: 13.28) with a minimum of $57.22, median of $2,030.76, mean of $2,373.98, and maximum of $25,533.00.

---

## 2. Data Quality Issues Identified & Solutions Applied

| Issue Identified | Location / Nature | Impact if Unaddressed | Engineering Solution |
| :--- | :--- | :--- | :--- |
| **Missing Weights** | 300 nulls (0.62%) in `weight` | Model execution failure or biased splits | Imputed using the training median (32,000 lbs), representing standard commercial trailer load mass. |
| **Missing Market Indices** | 374 nulls (0.78%) in `market_index` | Degradation of market signal | Imputed using training mean (1.0834), reflecting baseline neutral market conditions. |
| **Missing Spatial Coordinates in Inference Data** | `december_chart_inputs.csv` has 7 columns and lacks `pickup_lat/lon` and `delivery_lat/lon` | Defaulting to `(0, 0)` forces coordinates into the Atlantic Ocean, triggering out-of-domain tree leaves | Developed a city coordinate memory table during pipeline fitting that automatically maps city names to actual historical coordinates (e.g., Lexington: 38.09°N, 84.50°W; Fort Wayne: 41.08°N, 85.14°W). |
| **Temporal Saturation in Decision Trees** | Training data only spans months 1 to 10 (Jan–Oct) | Tree splits on `month` cannot extrapolate to December (month 12), causing all December predictions to collapse into an identical flat leaf | Implemented **cyclical annual encodings** (`sin_month`, `cos_month`) connecting December smoothly back to January, alongside explicit day-of-week demand offsets and an end-of-month volume indicator. |

---

## 3. Training & Validation Strategy (Data Split)

### Why Time-Based Splitting Was Selected
In logistics and rate forecasting, standard K-Fold random cross-validation introduces severe **data leakage** by using future load information to predict past loads. Because our final objective is to forecast unseen future months (November and December 2025), our validation strategy mirrored this exact scenario:

- **Training Split**: January 1, 2025 to September 30, 2025 (43,147 loads).
- **Holdout Validation Split**: October 1, 2025 to October 31, 2025 (4,853 loads).

The pipeline was fitted strictly on the Jan–Sep split, and evaluated on the out-of-time October holdout. After confirming validation metrics, the final ensemble was retrained on the complete 48,000-load dataset to maximize training sample density for November and December inference.

### Validation Performance Table (October Holdout)

| Model | RMSE ($) | MAE ($) | MAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| Ridge Regression (Baseline) | $653.37 | $135.83 | 7.79% | 0.8173 |
| LightGBM Regressor | $664.81 | $152.61 | 6.84% | 0.8108 |
| **Weighted Ensemble (85% LGBM + 15% Ridge)** | **$660.90** | **$143.43** | **6.51%** | **0.8131** |

---

## 4. Modeling Approach & Feature Engineering

### Feature Engineering
26 engineered features were constructed to capture domain relationships:
1. **Spatial & Geometric Features**:
   - `haversine_dist`: Great-circle geodesic distance in miles computed via spherical trigonometry.
   - `route_circuitousness`: Ratio of road distance to great-circle distance (`distance / haversine_dist`), highlighting non-direct routing.
2. **Market Signal Interactions**:
   - `quote_x_dist`: Interaction between initial broker quote signal and total mileage.
   - `market_x_dist`: Adjustment of total haul cost scaled by local regional market tightness.
   - `rate_per_mile_signal`: Normalized rate signal per mile.
   - `weight_x_dist`: Mass-distance work interaction.
3. **Categorical Encodings**:
   - `pickup_target_enc` & `delivery_target_enc`: Empirical Bayes smoothed target encoding for 64 origin and destination cities.
   - `equipment_code`: Ordinal cost encoding (Dry Van: 0, Flatbed: 1, Reefer: 2).
4. **Temporal Encodings**:
   - `sin_month`, `cos_month`: Continuous periodic representations of seasonal cycles.
   - `dow_effect`: Empirical day-of-week freight pricing offset.
   - `is_weekend`, `is_month_end`: High-volume dispatch indicators.

### Model Architecture: Ensemble Rationale
- **LightGBM** captures non-linear threshold effects, step-function pricing regimes, and complex feature interactions without requiring explicit polynomial expansion.
- **Ridge Regression** provides global linear regularization and smooth continuous temporal extrapolation.
- The **85/15 ensemble** achieved the lowest MAPE (**6.51%**) by combining the high-capacity interaction modeling of gradient boosting with the smooth extrapolation stability of linear regression.

---

## 5. Fixed December 2025 Prediction Analysis

The assessment evaluated a fixed route over all 31 days of December:
- **Route**: Lexington, KY to Fort Wayne, IN
- **Distance**: 360.0 miles
- **Equipment**: Dry Van
- **Weight**: 32,000 lbs

![Candidate December 2025 Predicted Load Rate](scorer_results/candidate_december.png)

### Behavioral Observations:
- **Plausible Rate Level**: The predicted rate averages **$831.55** over the 31 days. For a 360-mile haul, this corresponds to **$2.31/mile**, which closely aligns with the historical average of $2.22/mile plus standard regional adjustments for the Ohio Valley corridor.
- **Dynamic Weekly Fluctuations**: Predicted rates range between **$823.45 and $842.64** (standard deviation: $4.04), exhibiting midweek peaks on Wednesdays and lower dispatch pricing over weekends.
- **Absence of Artificial Flatlining**: The inclusion of learned city coordinates and cyclical calendar terms prevented the decision tree starvation issue, ensuring realistic temporal sensitivity throughout the holiday period.

---

## 6. Code Architecture Summary

- **`src/01_eda.py`**: Automated exploratory statistics, correlation matrices, and distribution diagnostics.
- **`src/02_feature_engineering.py`**: Modular `FeaturePipeline` class with persistent parameter state, coordinate dictionary memory, and leak-free transformation methods.
- **`src/03_train.py`**: End-to-end temporal validation harness and final production ensemble persistence (`models/model.pkl`).
- **`src/04_predict.py`**: Robust inference pipeline applying exact schemas for `validation_predictions.csv` and `december_chart_inputs.csv`.
- **`score.py`**: Official scorer script validating row counts, column orders, and visual chart generation.
