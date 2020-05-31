# -*- coding: utf-8 -*-

# Imports
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model

from utils import csv_to_dataset, history_points

# Data

ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('SPY_daily.csv')

test_split = 0.8
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)

# Model Definition

from tensorflow.keras import Sequential

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(history_points, 5)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x=ohlcv_train, y=y_train, epochs=100, batch_size=32)

model.save(f'basic_model_tf.h5')
model = load_model('basic_model_tf.h5')

# Testing, Prediction and Evaluation
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) +
                                                     np.abs(y_true))))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


assert unscaled_y_test.shape == y_test_predicted.shape
from sklearn.metrics import mean_squared_error

error = mean_squared_error(unscaled_y_test, y_test_predicted)
error3 = mean_absolute_percentage_error(unscaled_y_test, y_test_predicted)
smape = smape_kun(unscaled_y_test, y_test_predicted)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(12, 7, forward=True)

start = 0
end = -1

# alls = plt.plot(np.flip(unscaled_y), label='realw')
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.title(('LSTM S&P Price Prediction'))
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.savefig('lstm.png', dpi=500)

plt.show()
