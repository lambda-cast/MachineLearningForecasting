# PV Power Prediction

Predicts hourly power output (`PV1_Power_W` and `PV2_Power_W`) from weather conditions, using ~8 months of hourly sensor data (two rooftop PV panels).

## Folder structure

```
Input/    monthly raw CSV files (merged_2026-01.csv ... merged_2026-08.csv)
Output/   checkpoints, cleaned data, and trained model
```

## Notebooks (run in order)

| Notebook | What it does |
|---|---|
| `01_data_loading_overview.ipynb` | Loads all monthly files, provides data overview |
| `02_data_quality_feature_engineering.ipynb` | Data quality checks and feature engineering |
| `03_correlation_analysis.ipynb` | Correlation analysis between features |
| `04_visual_inspection.ipynb` | Visual inspection and sanity checks |
| `05_data_preparation_imputation.ipynb` | Missing data treatment and imputation |
| `06_Model_Training_and_Evaluation.ipynb` | Trains the model and evaluates it |

Each notebook reads a checkpoint file from `Output/` saved by the previous one — just run them top to bottom.

## The model

One **XGBoost** regressor trained on both panels at once (data is stacked so each timestamp appears twice — once per panel — with a `panel` flag telling the model which one it is). Predicting them together instead of separately gave the same accuracy with a simpler pipeline.

**Features used:** solar radiation, temperature, humidity, wind, pressure, PM2.5, hour of day, day of year, and the `panel` flag. Electrical readings (current, voltage, the other panel's power) are deliberately excluded — they're outputs, not things you'd know in advance.

**Accuracy (daylight hours, on unseen recent data):**

| Panel | MAE | RMSE | R² |
|---|---|---|---|
| PV1 | 14.4 W | 24.6 W | 0.964 |
| PV2 | 12.2 W | 22.4 W | 0.966 |

## Known limitations

- Only 8 months of data (Jan–Aug) — no autumn/winter yet, so accuracy may drop on unseen seasons
- PV1 consistently predicts slightly worse than PV2 across every model tried — likely a real physical difference (shading, soiling, orientation), worth checking on the hardware side
- Trained on one specific 2-panel installation — not yet validated on other sites/panel setups

## Using the trained model

```python
import joblib
import pandas as pd

model = joblib.load("Output/xgb_shared_model.joblib")
feature_cols = joblib.load("Output/feature_cols_shared.joblib")

# panel=0 for PV1, panel=1 for PV2 — same weather features either way
row = pd.DataFrame([{
    "Solar_Radiation_Wm2": 600, "Outdoor_Temp_C": 28, "Dew_Point_C": 15,
    "Wind_Speed_ms": 2.0, "Wind_Dir_deg": 180, "Pressure_hPa": 1015,
    "Humidity_pct": 45, "Rain_mm": 0, "PM25_ugm3": 12,
    "month": 7, "is_daylight": 1,
    "hour_sin": 0.5, "hour_cos": 0.87, "doy_sin": -0.3, "doy_cos": 0.95,
    "panel": 0,
}])[feature_cols]

predicted_power_w = model.predict(row)
```