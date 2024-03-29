# -*- coding: utf-8 -*-

# Imports
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data

from reporting import plot_dist

# Main MONTE CARLO
def main1():
    # download Apple price data into DataFrame
    spy = data.get_data_yahoo('SPY', start='1/1/2000 ')
    #### ESTIMATION OF THE MEAN AND VOLATILITY #####
    # calculate the compound annual growth rate ( CAGR ) which
    # will give us our mean return input (mu)
    days = (spy.index[-1] - spy.index[0]).days
    cagr = ((((spy['Adj Close'][-1]) / spy['Adj Close'][1])) **
            (365.0 / days)) - 1
    print('CAGR =', str(round(cagr, 4) * 100) + "%")
    mu = cagr
    # create a series of percentage returns and calculate the annual volatility of returns
    spy['Returns'] = spy['Adj Close'].pct_change()
    vol = spy['Returns'].std() * math.sqrt(252)
    print(" Annual Volatility =", str(round(vol, 4) * 100) + "%")
    #### MONTE CARLO ####
    # Define Variables
    S = spy['Adj Close'][-1]  # starting stock price (i.e. last available real stock price )
    T = 2500  # Number of trading days
    prices = pd.DataFrame()
    plt.figure(figsize=(12, 7))
    for i in range(300):
        # create list of daily returns using random normal distribution
        daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
        # set starting price and create price series generated by above random daily returns
        price_list = [S]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)
        # plot data from each individual run which we will plot at the end
        prices[i] = (price_list)
        plt.plot(price_list)
    # show the plot of multiple price series created above
    plt.xlabel('Days')
    plt.ylabel('Generated Price')
    plt.title('Monte Carlo Simulation of S&P Prices')
    #plt.savefig('images/montecarlo.png', dpi=500)
    plt.show()
    plot_dist(prices, title='Monte Carlo Simulation Log-Returns Distribution', fname='MCDist', save=False)
    return prices


def generate_prices(n):
    list1 = main1().iloc[:, 0:n]
    return list1
