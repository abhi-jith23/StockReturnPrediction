# Stock Next-Day Returns Prediction (LSTM + GRU)

**Status:** 🚧 **Ongoing research** (pipeline is stable; results + hyperparameter tuning are still being iterated).

This repository implements a **leakage-aware**, end-to-end workflow to forecast **next-day log returns** for liquid US equities using **daily Yahoo Finance data** (via `yfinance`).  
It compares strong baselines (naive, ARIMA/ARIMAX, XGBoost) against a **stacked LSTM→GRU** sequence model trained on **PCA-transformed engineered features**.

> ⚠️ **Not financial advice / not for trading.** This is a research/educational project using third-party market data.

---

## Project Goals

1. Build a **reproducible** pipeline for **next-day log return** prediction.
2. Ensure **time-series correctness** (chronological splits, training-only preprocessing, no future leakage).
3. Compare:
   - Naive baseline
   - ARIMA / ARIMAX
   - XGBoost (engineered features)
   - XGBoost (PCA)
   - **LSTM + GRU** (PCA features, 60-day lookback)
4. Track experiments with **CSV logging** for traceability (Optuna initial exploration + Grid Search for reproducibility).

---

## Data

### Tickers
**Targets (equities):**
- `AAPL`, `MSFT`, `GOOG`, `AMZN`

**Exogenous / market series:**
- `SPY`, `QQQ`, `XLK`, `^VIX`, `^TNX`, `UUP`, `CL=F`, `GC=F`

### Date Range
- **2013-01-01 → 2026-01-26** (daily frequency; trading days)

### Target Definition
Let \( P_t \) be **Adjusted Close**.
- **Log return:** `r_t = log(P_t / P_{t-1})`
- **Supervised target:** `y_t = r_{t+1}` (one-step forward shift)

Rows with missing **Adjusted Close** for target tickers are removed.

---

## Feature Engineering (High-Level)

All features are computed per ticker after sorting by date.

### Endogenous (per target ticker)
- Returns: log returns + simple returns
- Range proxies: `(High-Low)/Close`, `(Close-Open)/Open`
- Rolling stats: mean & volatility windows `{5, 10, 20, 60}`
- Trend indicators: SMA/EMA, MACD family
- Oscillators/bands: RSI(14), Bollinger width (20, 2σ)
- Volume: % change, 20D z-score, OBV
- Calendar: day-of-week, month, month-end
- Lag features: endogenous return lags **1..60**

### Exogenous (wide join by date)
For each (targets + exogenous series), compute **log return lags 1..20**, pivot wide, then left-join onto the target panel by date.  
Self-duplicates (lag-1..20 terms already covered by 1..60) are removed.

**Modeling panel (current):**
- ~**13,140 rows** and **346 columns** (combined across 4 target tickers)
- First ≥ 60 observations per ticker are naturally dropped due to lag/window requirements

---

## Dimensionality Reduction (PCA)

For PCA-based models:
- Standardize features using **training-only** statistics
- Apply PCA to retain **80% explained variance**
- Current setup yields **~91 PCA components**

---

## Models Implemented

### Baselines
- **Naive:** \(\hat r_{t+1} = 0\)
- **ARIMA:** univariate ARIMA fit on training target series; forecast over test horizon
- **ARIMAX:** ARIMA with exogenous regressors (top-k engineered features; k=20)
- **XGBoost:** trained on engineered features
- **XGBoost (PCA tuned):** trained on PCA-transformed standardized features

### Deep Learning: LSTM + GRU (PCA)
- Input: PCA features (d ≈ 91)
- Lookback window: **60 trading days**
- Architecture: `LSTM(h1) → GRU(h2) → Dense(d1) → Dense(d2) → Output(1)`
- Training:
  - Loss: MSE
  - Optimizer: AdamW
  - Gradient clipping: max-norm = 1.0
  - Early stopping on validation RMSE (max 50 epochs; patience 7)
  - Temporal train/val split within training sequences: **90/10** (last 10% = validation)

### Hyperparameter Search
- **Optuna (initial):** used early for fast exploration (with pruning). Not the final reproducibility path.
- **Grid Search (current):** explicit search grid with **per-run CSV logging** to ensure traceability and resume support.

---

## Evaluation Protocol (Leakage-Safe)

- Chronological split:
  - **Train:** first 80%
  - **Test:** last 20%
- Preprocessing (scaler + PCA) fit on **train only**, then applied forward
- For sequence test construction:
  - prepend the last **L=60** training rows to test features to create complete test sequences
  - targets remain strictly in the test period

### Metrics
- MAE, RMSE, \(R^2\)
- Directional accuracy (MDA):
  \[
  \mathrm{MDA}=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\big(\mathrm{sign}(r_i)=\mathrm{sign}(\hat r_i)\big)
  \]

---

## Current Results Snapshot (AAPL)

> **Note:** Next-day return prediction is extremely noisy; many models cluster tightly around strong baselines.

| Model | MAE | RMSE | R² | MDA |
|---|---:|---:|---:|---:|
| Naive | 0.010954 | 0.016341 | -0.001397 | 0.456452 |
| ARIMA | 0.010946 | 0.016338 | -0.001065 | 0.520968 |
| ARIMAX | 0.011260 | 0.016579 | -0.030851 | 0.522581 |
| XGBoost | 0.011975 | 0.017424 | -0.138567 | 0.496774 |
| XGBoost_PCA_Tuned | 0.011075 | 0.016321 | 0.000952 | 0.529032 |
| LSTM_GRU_Optuna | **0.010902** | **0.016297** | **0.004003** | **0.530645** |
| LSTM_GRU_GridSearch | 0.011051 | 0.016352 | -0.002766 | 0.522581 |

---

## Repository Structure

```

.
├── data
│   ├── raw
│   │   ├── yfinance_ohlcv_long.csv
│   │   └── yfinance_ohlcv_long.parquet
│   └── processed
│       ├── engineered_long_all_tickers.parquet
│       ├── modeling_panel_targets.parquet
│       └── phase2_artifacts
│           ├── pca.joblib
│           ├── scaler.joblib
│           ├── metrics.csv
│           ├── grid_best_params.json
│           ├── phase2_gridsearch_runs.csv
│           ├── lstm_gru_pca_weights.pt
│           └── test_predictions.csv
└── notebooks
├── phase_1
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_feature_engineering.ipynb
│   └── 03_baselines_and_metrics.ipynb
└── phase_2
│   ├── 00_info.ipynb
│   ├── 01_lstm_gru_next_day_optuna.ipynb
│   ├── 02_lstm_gru_next_20_grid_search_r2.ipynb
│   ├── 03_lstm_gru_next_20_grid_search_rmse.ipynb
│   └── 04_lstm_gru_next_day_grid_search_rmse.ipynb

```

---

## Known Limitations (Current Research Focus)

* **Hyperparameter sensitivity:** An early Optuna run produced the best deep model score, but exact hyperparameters were not persisted due to technical issues (non-reproducible run).
  Grid search improves traceability but has not yet recovered that best score consistently.
* **Results currently reported for AAPL first.** The pipeline is designed to extend to `MSFT/GOOG/AMZN` once tuning is stabilized.

---

## Roadmap (Planned)

* Stabilize hyperparameter selection + deterministic runs
* Extend identical evaluation to `MSFT`, `GOOG`, `AMZN`
* Add rolling-origin evaluation (time-series cross-validation)
* Explore attention-based time-series models (e.g., Transformer variants)
* Add richer covariates (e.g., news/sentiment; macro indicators)

---

## Data & Legal Disclaimer

* `yfinance` is an open-source wrapper around Yahoo’s public endpoints and is intended for **research/educational** use.
* Yahoo Finance market data is provided **“as is”** for **informational purposes** and is **not intended for trading/investing**.
* Do not redistribute data in ways that violate the upstream provider’s terms.

See:

* `yfinance` documentation: [https://ranaroussi.github.io/yfinance/](https://ranaroussi.github.io/yfinance/)
* Yahoo Finance help (data provider notes): [https://help.yahoo.com/kb/SLN2310.html](https://help.yahoo.com/kb/SLN2310.html)

---

## Author

**Abhijith Senthilkumar**
MSc Data Science, University of Luxembourg

---

## License

License is **TBD** 

---
