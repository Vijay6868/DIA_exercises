import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


import statsmodels.api as sm

#load the dataset
data = sm.datasets.co2.load_pandas()
co2 = data.data

#handle missing values
co2 = co2.fillna(co2.interpolate())

#set the index to datetime
co2.index = pd.to_datetime(co2.index)

#decompose the time series
decomposition =  seasonal_decompose(co2['co2'], model='multiplicative', period=12)

#extract the componenets 
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#plot the decomposed model 
# Plot the decomposed components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(co2['co2'], label='Original', color='blue')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='green')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals', color='black')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
