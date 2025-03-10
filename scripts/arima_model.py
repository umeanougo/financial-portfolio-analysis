import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load and prepare data
vfv_data = pd.read_csv('../data/vfv_raw_data.csv', parse_dates=['Date'], index_col='Date')
vfv_data.sort_index(inplace=True)
vfv_data.rename(columns={'Market price (CAD)': 'Close'}, inplace=True)
vfv_data['Close'] = vfv_data['Close'].replace('[\$,]', '', regex=True).astype(float)

vfv_data = vfv_data.asfreq('B')  # 'B' means business day frequency
vfv_data['Close'] = vfv_data['Close'].interpolate(method='time')  # Fill any missing dates if needed

# Feature Engineering
vfv_data['Daily_Return'] = vfv_data['Close'].pct_change()
vfv_data['SMA50'] = vfv_data['Close'].rolling(window=50).mean()
vfv_data['SMA200'] = vfv_data['Close'].rolling(window=200).mean()
vfv_data['Volatility'] = vfv_data['Daily_Return'].rolling(window=30).std()
vfv_data['Momentum'] = vfv_data['Close'].diff(10)

# Inspect Data
print(vfv_data.info())
print(vfv_data.describe())

# Prepare the data
X = vfv_data.drop(columns=['NAV (CAD)', 'Close']).dropna()
y = vfv_data['Close'].loc[X.index]

# Now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(len(X_test), len(y_test))  # They should match now

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ARIMA Model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_result = arima_model.fit()
forecast = arima_result.forecast(steps=len(X_test))
forecast_index = pd.date_range(start=y_train.index[-1], periods=len(X_test), freq='B')

print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_test: {len(y_test)}")
print(f"Length of y_pred_lr: {len(y_pred_lr)}")
print(f"Length of forecast: {len(forecast)}")

plt.figure(figsize=(10, 6))
plt.plot(vfv_data.index, vfv_data['Close'], label='Actual VFV Close Price')
plt.plot(forecast_index, forecast, 'r--', label='ARIMA Forecast', linewidth=2.5)
plt.legend()
plt.title('VFV Close Price Forecast')
plt.show()

# Metrics
print("Linear Regression Metrics:")
rmse = root_mean_squared_error(y_test, y_pred_lr)
print("RMSE:", rmse)
print("R^2:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Metrics:")
rmse = root_mean_squared_error(y_test, y_pred_rf)
print("RMSE:", rmse)
print("R^2:", r2_score(y_test, y_pred_rf))