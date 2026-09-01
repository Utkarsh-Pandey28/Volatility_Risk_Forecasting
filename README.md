# Financial Volatility Forecasting & Market Risk Estimation

A time-series project that forecasts financial volatility, estimates market risk, and predicts next-day stock prices using both classical statistical models and machine learning.

## Objective

1. **Forecast financial volatility and estimate market risk** using historical price and trading data.
2. **Predict tomorrow's stock price.**
3. Answer: *"What is the probability that the market will enter a high-volatility regime over the next 5 trading days?"*

## Dataset

`Core_TimeSeries.csv` — synthetic daily OHLC + volume + technical-indicator data (30,000 rows).

> Note: The dataset is not included in this repo (see [Data](#data) below on how to add it).

## Project Roadmap

| Step | What we do |
|------|------------|
| 1 | Load & clean data |
| 2 | Daily Returns, Log Returns |
| 3 | Rolling Volatility, Historical Volatility |
| 4 | VaR, CVaR (historical + parametric) |
| 5 | Maximum Drawdown, Sharpe Ratio |
| 6 | ARIMA — price forecasting |
| 7 | SARIMAX — price forecasting with exogenous features |
| 8 | GARCH(1,1) — volatility forecasting |
| 9 | Random Forest — next-day price prediction |
| 10 | XGBoost — next-day price prediction |
| 11 | LSTM — next-day price prediction |
| 12 | Volatility regime classification (5-day-ahead) |



## Key Techniques

- **Risk metrics:** Daily/Log Returns, Rolling & Historical Volatility, Value at Risk (VaR), Conditional VaR (CVaR), Maximum Drawdown, Sharpe Ratio
- **Statistical forecasting:** ARIMA, SARIMAX (with volume as an exogenous regressor), GARCH(1,1) for conditional volatility
- **Machine learning:** Random Forest Regressor, XGBoost Regressor, LSTM (Keras/TensorFlow) for next-day close price prediction
- **Classification:** Random Forest classifier to predict high-volatility regimes over a 5-day forward horizon, cross-checked against GARCH forecasts

## Results Summary

The notebook compares all price-forecasting models on **MAE** and **RMSE**, and produces a next-day price forecast from each. It also outputs a probability estimate for entering a high-volatility regime in the next 5 trading days, using both an ML classifier and a GARCH-based cross-check.

> **Note on the regime classifier:** the time-based train/test split leaves the "High Vol" class under-represented in the test set, which inflates overall accuracy while understating recall for that class. For production use, prefer walk-forward validation and report precision/recall/F1 for the high-vol class specifically — see the notebook's caveats section for details.

## Repository Structure
