#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:53:51 2020

@author: oscar
"""
import numpy as np
from numpy import hstack
from montecarlo import generate_prices


# generate n real samples with class labels
def generate_real_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = np.arange(n)
    X1 = X1 * np.pi / 180
    # generate outputs X^2
    X2 = np.cos(X1)
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = np.ones((n, 1))
    return X, y

def generate_monte_carlo(n):
     # generate inputs in [-0.5, 0.5]
    X1 = np.arange(253)
    # generate outputs X^2
    X2 = generate_prices(n).values
    # stack arrays
    X1 = X1.reshape(253,1)
    X2 = X2.reshape(253, n)
    X = hstack((X1, X2))
    # generate class labels
    y = np.ones((n, 1))
    return X, y

# generate n real samples with class labels
def gaussian_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = np.arange(n)
    X1 = X1
    # generate outputs X^2
    X2 = np.random.normal(0,0.1,n)
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = np.ones((n, 1))
    return X, y


import pandas as pd


def generate_sp_samples(n):
    data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
    X1 = np.arange(len(data[0:n]))
    # X2 = np.log(data) - np.log(data.shift(1))
    X2 = data[0:n].values
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y

def generate_double_samples(n):
    data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
    data2 = pd.read_csv('AAPL_daily.csv')[['date', '4. close']].set_index('date').sort_index()
    X1 = np.arange(len(data[0:n]))
    # X2 = np.log(data) - np.log(data.shift(1))
    # X2 = data[0:n].values
    # X2 = data[0:n].values
    X = pd.concat([data,data2], axis=1).dropna()[:n].values.reshape(n,2)
    y = np.ones((n, 1))
    return X, y

def generate_log_sp_samples(n):
    data = pd.read_csv('./SPY_daily.csv')[['date', '4. close']].set_index('date').sort_index()
    X1 = np.arange(len(data[0:n]))
    # X2 = np.log(data) - np.log(data.shift(1))
    X2 = np.log(data[0:n].values)
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y



def generate_ts_samples(n):
    data = pd.read_csv("AirPassengers.csv")[["#Passengers"]]
    X1 = np.arange(len(data[0:n]))
    X2 = data[0:n].values
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y


# generate n fake samples with class labels
def generate_fake_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = np.rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = np.zeros((n, 1))
    return X, y
