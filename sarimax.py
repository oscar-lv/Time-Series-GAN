#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:36:30 2020

@author: oscar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from itertools import product

# Parameter definition
ps = range(0,5)
qs = range(0,5)
PS = range(0,5)
QS = range(0,5)
d = 1 
D = 1 
s = 5

parameters = product(ps,qs,PS,QS)
parameters_list = list(parameters)

len(parameters_list)



df = pd.read_csv("SPY_daily.csv").fillna(0).set_index('date').sort_index()
df.head()

plt.figure(figsize=(10, 10))
lag_plot(df['4. close'], lag=5)
plt.title('SP Autocorrelation plot')



train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]
plt.figure(figsize=(12, 7))
plt.title('S&P Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df['4. close'], 'blue', label='Training Data')
plt.plot(test_data['4. close'], 'green', label='Testing Data')
plt.xticks(np.arange(0, 7982, 1300), df.index[0:7982:1300])
plt.legend()


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) +
                                                     np.abs(y_true))))


train_ar = train_data['4. close'].values
test_ar = test_data['4. close'].values
history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in tqdm(range(len(test_ar))):
    model = SARIMAX(history, order=(0,1,0), season_order=(2,1,4,5))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)

plt.figure(figsize=(12, 7))
plt.plot(df['4. close'], 'green', color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',
         label='Predicted Price')
plt.plot(test_data.index, test_data['4. close'], color='red', label='Actual Price')
plt.title('S&P Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0, 7982, 1300), df.index[0:7982:1300])
plt.legend()
# plt.savefig('arima5_1_0.png',dpi=500)
