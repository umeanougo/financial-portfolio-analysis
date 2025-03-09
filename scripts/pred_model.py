import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load and prepare data
vfv_data = pd.read_csv('../data/vfv_raw_data.csv', parse_dates=['Date'], index_col='Date')
vfv_data.sort_index(inplace=True)
vfv_data.rename(columns={'Market price (CAD)': 'Close'}, inplace=True)
vfv_data['Close'] = vfv_data['Close'].replace('[\$,]', '', regex=True).astype(float)

# Feature engineering
vfv_data['Returns'] = vfv_data['Close'].pct_change()
vfv_data.dropna(inplace=True)

# Train-test split
X = vfv_data[['Returns']].shift(1).dropna()
y = vfv_data['Returns'].dropna()

# Ensure they're both Series or DataFrame before aligning
X = pd.DataFrame(X)  # Make sure X is a DataFrame
y = pd.Series(y)     # Make sure y is a Series

# Now align them
X, y = X.align(y, join='inner', axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluate Linear Regression
print("Linear Regression Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr))}")
print(f"R^2: {r2_score(y_test, y_pred_lr)}")

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf))}")
print(f"R^2: {r2_score(y_test, y_pred_rf)}")

# ARIMA Model for time series forecasting
model = ARIMA(vfv_data['Close'], order=(5,1,5))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# Plot forecast
plt.figure(figsize=(10,6))
plt.plot(vfv_data['Close'], label='Actual VFV Close Price')
plt.plot(pd.date_range(vfv_data.index[-1], periods=30, freq='B'), forecast, label='ARIMA Forecast', linestyle='--')
plt.legend()
plt.title('VFV Close Price Forecast')
plt.show()