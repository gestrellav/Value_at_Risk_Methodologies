#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:18:27 2024

@author: george
"""

# Library
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Contents
# 1. Defining Value at Risk
## 1.a. Significance Level and Confidence Level
## 1.b. Risk Horizon
# 2. How to calculate Daily VaR 

data = yf.download('MSFT', '2010-01-01', '2024-12-30')['Close']
data = data.reset_index()
data.rename(columns={'Close':'Price'}, inplace=True)
data['Daily_Log_Return'] = np.log(data['Price']/data['Price'].shift(1))

# Daily Histogram and Historical Value at Risk
## Histogram
plt.hist(data['Daily_Log_Return'].dropna(), bins=30,alpha=0.5, label='Microsoft Inc.', color='orange', edgecolor='orange')
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frecuency')
plt.legend()
plt.show()
## Daily Value at Risk
Daily_HVaR = np.percentile(data['Daily_Log_Return'].dropna(), (1 - 0.995) * 100)

# Weekly Histogram and Historical Value at Risk
data['Weekly_Log_Return'] = np.log(data['Price']/data['Price'].shift(5))
plt.hist(data['Weekly_Log_Return'].dropna(), bins=30, alpha=0.6, label='Microsoft Inc.', color='skyblue', edgecolor='skyblue')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frecuency')
plt.legend()
plt.show()

Weekly_HVaR = np.percentile(data['Weekly_Log_Return'].dropna(), (1 - 0.995) * 100)

# Merge Chart
plt.hist(data['Daily_Log_Return'].dropna(), bins=30,alpha=0.5, label='Daily Log Returns', color='blue', edgecolor='blue')
plt.hist(data['Weekly_Log_Return'].dropna(), bins=30, alpha=0.6, label='Weekly Log Returns', color='red', edgecolor='red')
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frecuency')
plt.legend()
plt.show()


# Function
def historicalVaR(tickers, start_date, end_date, var_days_list, confidence_level):
    # Close Prices
    prices = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Conditional for one Ticker
    if isinstance(prices, pd.Series) and len(tickers) == 1:
        prices = prices.to_frame(name=tickers[0])
    
    # DataFrame
    h_var_dict = {ticker: {} for ticker in tickers}
    
    for var_days in var_days_list:
        rets = np.log(prices / prices.shift(var_days))
        for ticker in rets.columns:
            var = np.percentile(rets[ticker].dropna(), (1 - confidence_level) * 100)
            h_var_dict[ticker][f'HVaR_{var_days}'] = var
    
    # To Dictionary to DataFrame
    h_var_df = pd.DataFrame.from_dict(h_var_dict, orient='index').reset_index()
    h_var_df.rename(columns={'index': 'Ticker'}, inplace=True)
    h_var_df.sort_values(by='Ticker', ascending=True, inplace=True)
    h_var_df.reset_index(drop=True, inplace=True)
    
    return h_var_df

# Weekly Historical Value at Risk
tickers = ['AMZN', 'AAPL', 'NVDA', 'TSLA', 'INTC', 'META', 'MSFT', 'GOOGL', 'JPM', 'NFLX', 'COST', 'BAC']
start_date = '2010-01-01'
end_date = '2024-12-30'
var_days = [1, 5, 22, 252]
confidence_level = 0.995

h_var = historicalVaR(tickers, start_date, end_date, var_days, confidence_level)
