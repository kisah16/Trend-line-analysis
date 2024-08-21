# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:42:46 2024

@author: kisah
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices,
    # return negative val if invalid

    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):

    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y)

    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step  # current step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0)  # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases.
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            # If increasing by a small amount fails,
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:  # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0:  # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else:  # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # slope failed/didn't reduce error
            curr_step *= 0.5  # Reduce step size
        else:  # test slope reduced error
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared)
    # coefs[0] = slope,  coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def weighted_average(values, weights):
    if len(values) != len(weights):
        raise ValueError("Length of values and weights should be the same")

    weighted_sum = sum(value * weight for value,
                       weight in zip(values, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")

    return weighted_sum / total_weight


symbol_list = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','AVAXUSDT','ADAUSDT','XRPUSDT','TRXUSDT','DOTUSDT','MATICUSDT','LINKUSDT',
               'ICPUSDT','UNIUSDT','OPUSDT','FILUSDT','APTUSDT','MKRUSDT','GRTUSDT','AAVEUSDT','FTMUSDT',
               'DOGEUSDT','LTCUSDT','ATOMUSDT','XLMUSDT','NEARUSDT','XMRUSDT','LDOUSDT','IMXUSDT','FLOWUSDT','XTZUSDT','CHZUSDT']
all_data = []

for symbol in symbol_list:
    print(symbol)
    col_names = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime',
                 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore']
    df = pd.read_csv('C:/Users/kisah/Desktop/data_all_1hour/' + symbol + "1hourall.csv", names=col_names, header=None)
    df['Date'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    df['Timestamp'] = df['Date']
    df = df.drop(columns=['Date', 'closeTime', 'quoteAssetVolume',
                 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df = df.drop(df.head(1).index)
    
    df.set_index('Timestamp', inplace=True)

    data = df.copy()

    low_slope_all = []
    high_slope = [np.nan] * len(data)
    high_slope_all = []
    low_slope = [np.nan] * len(data)
    for a in range(0,25):
        
        low_slope_all.append(low_slope.copy())
        high_slope_all.append(high_slope.copy())
    
    for data_iterator in range(120, len(data)):
        try:
            data_to_use = data[:data_iterator + 1].copy()
        

            for lookback in range(0,25):
                candles = data_to_use[len(data_to_use) - (lookback * 5 + 5):]
                support_coefs, resist_coefs = fit_trendlines_high_low(candles['High'],
                                                                          candles['Low'],
                                                                          candles['Close'])
        
                low_slope_all[lookback][data_iterator] = support_coefs[0]
                high_slope_all[lookback][data_iterator] = resist_coefs[0]

        
        except:
            print(data_iterator)
        
    data.reset_index(inplace=True)
    for a in range(0,25):
        
        data['low_slope_' + str(a * 5 + 5)] = low_slope_all[a]/data['Close']
        data['high_slope_' + str(a * 5 + 5)] = high_slope_all[a]/data['Close']
    
    data.to_csv("C:/Users/kisah/Desktop/data_with_slope/1hour/" + symbol + ".csv")
        
        
        
   
