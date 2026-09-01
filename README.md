Financial Volatility Forecasting & Market Risk Estimation

A time-series project that forecasts financial volatility, estimates market risk, and predicts next-day stock prices using both classical statistical models and machine learning.

Objective
Forecast financial volatility and estimate market risk using historical price and trading data.
Predict tomorrow's stock price.
Answer: "What is the probability that the market will enter a high-volatility regime over the next 5 trading days?"
Dataset

Core_TimeSeries.csv — synthetic daily OHLC + volume + technical-indicator data (30,000 rows).

Note: The dataset is not included in this repo (see Data below on how to add it). Screenshots below were generated from a representative sample run for illustration.

Project Roadmap
Step	What we do
1	Load & clean data
2	Daily Returns, Log Returns
3	Rolling Volatility, Historical Volatility
4	VaR, CVaR (historical + parametric)
5	Maximum Drawdown, Sharpe Ratio
6	ARIMA — price forecasting
7	SARIMAX — price forecasting with exogenous features
8	GARCH(1,1) — volatility forecasting
9	Random Forest — next-day price prediction
10	XGBoost — next-day price prediction
11	LSTM — next-day price prediction
12	Volatility regime classification (5-day-ahead)
Screenshots
Daily & Log Returns

Show Image

Rolling Volatility vs. Historical Volatility

Show Image

Value at Risk (VaR) — Return Distribution

Show Image

Maximum Drawdown

Show Image

ARIMA — Price Forecast

Show Image

GARCH(1,1) — Conditional Volatility

Show Image

Random Forest — Feature Importance

Show Image

Random Forest — Actual vs Predicted Close

Show Image

Model Comparison (Test RMSE)

Show Image

Key Techniques
Risk metrics: Daily/Log Returns, Rolling & Historical Volatility, Value at Risk (VaR), Conditional VaR (CVaR), Maximum Drawdown, Sharpe Ratio
Statistical forecasting: ARIMA, SARIMAX (with volume as an exogenous regressor), GARCH(1,1) for conditional volatility
Machine learning: Random Forest Regressor, XGBoost Regressor, LSTM (Keras/TensorFlow) for next-day close price prediction
Classification: Random Forest classifier to predict high-volatility regimes over a 5-day forward horizon, cross-checked against GARCH forecasts
Results Summary

The notebook compares all price-forecasting models on MAE and RMSE, and produces a next-day price forecast from each. It also outputs a probability estimate for entering a high-volatility regime in the next 5 trading days, using both an ML classifier and a GARCH-based cross-check.

Note on the regime classifier: the time-based train/test split leaves the "High Vol" class under-represented in the test set, which inflates overall accuracy while understating recall for that class. For production use, prefer walk-forward validation and report precision/recall/F1 for the high-vol class specifically — see the notebook's caveats section for details.

Repository Structure
.
├── Volatility_Risk_Forecasting_TS.ipynb   # Main analysis notebook
├── Core_TimeSeries.csv                    # Input dataset (add locally, see Data section)
├── requirements.txt                       # Python dependencies
├── images/                                # Chart screenshots used in this README
└── README.md
Requirements
Python 3.9+
Key libraries: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, xgboost, statsmodels, arch, tensorflow

Install everything with:

bash
pip install -r requirements.txt
Data

Place Core_TimeSeries.csv in the project root before running the notebook. The file is expected to have (at minimum) the following columns: Date, Open_Price, High_Price, Low_Price, Close_Price, Volume, Market_Cap, Volatility_Range, SMA_20, SMA_50, RSI_14.

Usage
bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook Volatility_Risk_Forecasting_TS.ipynb
Possible Extensions
Grid-search / auto_arima-style order selection for ARIMA & SARIMAX instead of fixed (p,d,q)
EGARCH/GJR-GARCH to capture asymmetric volatility (leverage effect)
Walk-forward (rolling-origin) validation for the regime classifier
Stratified sampling by volatility regime when splitting train/test data
