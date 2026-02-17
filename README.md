# Value-at-Risk Forecasting with Realized Volatility Measures (SPY)

Forecast **one-day-ahead Value-at-Risk (VaR)** for **SPY daily log-returns** at confidence levels **1%, 5%, 10%**, using **intraday-based volatility measures** (realized volatility `rv5` and bipower variation `bv`) and comparing three model families:
- **GARCH** (with heavy-tailed innovations)
- **HAR-RV** (multi-horizon realized volatility with optional jump proxy)
- **Quantile Regression** (direct conditional quantile modeling)

This repository contains the code/notebooks to reproduce the analysis, feature engineering, model selection, and backtesting described in the accompanying report. 

---

## Project highlights

- Documents **volatility clustering** and **long-memory** patterns via ACF plots of `|log_ret|` and `log(rv5)`. 
- Builds interpretable **multi-horizon volatility features** (daily/weekly/monthly) from `log(rv5)`. 
- Adds an optional **jump proxy**: `J_t = max(rv5_t - bv_t, 0)` to capture discontinuous moves.
- Uses a realistic, no-look-ahead **chronological split**: first **80% train**, last **20% test**, with **rolling forecasting** during tuning.  
- Evaluates VaR forecasts with:
  - **Empirical coverage** (violation rate)
  - **Quantile (pinball) loss**
  - **Kupiec unconditional coverage test**
  - A **composite score** for hyperparameter selection across the 3 alphas.

---

## Results

All three approaches pass the **Kupiec test at 5% significance** on the test set for **α ∈ {1%, 5%, 10%}**, with overall performance favoring **HAR-RV** in this project’s composite ranking. For more, please check the report : 



