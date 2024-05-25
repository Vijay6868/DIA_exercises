import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = sm.datasets.co2.load_pandas()
co2 = data.data

# Handle missing values
co2 = co2.fillna(co2.interpolate())

# Set the index to datetime
co2.index = pd.to_datetime(co2.index)

# Decompose the time series
decomposition = seasonal_decompose(co2['co2'], model='multiplicative', period=12)
co2['trend'] = decomposition.trend
co2['seasonal'] = decomposition.seasonal
co2['residual'] = decomposition.resid

# Split the data into training and testing sets
train_size = int(len(co2) * 0.8)
train, test = co2[:train_size], co2[train_size:]

# Forecasting using the trend and seasonal components
# Extract trend and seasonal components for training data
train_trend = train['trend']
train_seasonal = train['seasonal']

# Forecast future trend values
future_trend = train_trend.dropna().iloc[-1] + (train_trend.dropna().diff().mean() * len(test))

# Forecast future seasonal values
# Use a cyclic pattern for the seasonal component
future_seasonal = pd.concat([train_seasonal.dropna().iloc[-12:]] * (len(test) // 12 + 1))[:len(test)]
future_seasonal.index = test.index

# Combine components to create the forecast
forecast = future_trend + future_seasonal

# Evaluate the model
mae = mean_absolute_error(test['co2'], forecast)
mse = mean_squared_error(test['co2'], forecast)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Plot the actual data and the forecast
plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(co2.index, co2['co2'], label='Actual', color='blue')

# Plot forecasted values
plt.plot(test.index, forecast, label='Forecast', color='red')

plt.xlabel('Date')
plt.ylabel('CO2 Levels')
plt.title('CO2 Forecast')
plt.legend()
plt.show()
