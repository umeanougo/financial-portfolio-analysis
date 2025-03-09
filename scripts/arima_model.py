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
vfv_data['Close'].interpolate(method='time')  # Fill any missing dates if needed

# Inspect Data
print(vfv_data.info())
print(vfv_data.describe())

# Prepare the data
X = np.arange(len(vfv_data)).reshape(-1, 1)
y = vfv_data['Close'].values

# Align X and y to remove any potential mismatched dates
X, y = vfv_data.index, vfv_data['Close'].dropna()
X = np.arange(len(y)).reshape(-1, 1)  # Match length of y after dropping NaNs

# Now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(len(X_test), len(y_test))  # They should match now

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Remove NaNs from both X_train and y_train
mask = ~np.isnan(y_train)
X_train = X_train[mask]
y_train = y_train[mask]

print(f"Remaining NaNs in y_train: {np.isnan(y_train).sum()}")  # Should be 0 now

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ARIMA Model
arima_model = ARIMA(vfv_data['Close'], order=(5,1,0))
arima_result = arima_model.fit()
forecast = arima_result.forecast(steps=len(X_test))
forecast_index = pd.date_range(start=vfv_data.index[-len(X_test)], periods=len(X_test), freq='B')

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

# Remove NaNs from X_test and y_test
mask_test = ~np.isnan(y_test)
X_test = X_test[mask_test]
y_test = y_test[mask_test]

print(f"Remaining NaNs in y_test: {np.isnan(y_test).sum()}")  # Should be 0 now

print(y_test.shape)
print(y_pred_lr.shape)
print(forecast.shape)

# Metrics
print("Linear Regression Metrics:")
# print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
rmse = root_mean_squared_error(y_test, y_pred_lr)
print("RMSE:", rmse)
print("R^2:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Metrics:")
#print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
rmse = root_mean_squared_error(y_test, y_pred_rf)
print("RMSE:", rmse)
print("R^2:", r2_score(y_test, y_pred_rf))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(vfv_data.index, vfv_data['Close'], label='Actual VFV Close Price')
plt.plot(vfv_data.index[-len(X_test):], forecast, 'r--', label='ARIMA Forecast', linewidth=2.5)
plt.legend()
plt.title('VFV Close Price Forecast')
plt.show()
