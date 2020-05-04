#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:28:22 2019

@author: oscar
"""

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import json
import argparse


def save_dataset(symbol, time_window):
    #credentials = json.load(open('creds.json', 'r'))
    #api_key = credentials['av_api_key']
    api_key = 'EXLBHNQT771YZW01'
    print(symbol, time_window)
    ts = TimeSeries(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data, meta_data = ts.get_intraday(
            symbol=symbol, interval='1min', outputsize='full')
    elif time_window == 'daily':
        data, meta_data = ts.get_daily(symbol, outputsize='full')
    elif time_window == 'daily_adj':
        data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
    
    pprint(data.head(10))

    data.to_csv(f'./{symbol}_{time_window}.csv')
    return data, meta_data

df, md = save_dataset('AAPL', 'daily')
df2, md2= save_dataset('SPY', 'intraday')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('symbol', type=str, help="the stock symbol you want to download")
    parser.add_argument('time_window', type=str, choices=[
                        'intraday', 'daily', 'daily_adj'], help="the time period you want to download the stock history for")

    namespace = parser.parse_args()
    save_dataset(**vars(namespace))