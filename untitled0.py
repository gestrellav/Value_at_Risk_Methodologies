#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:04:44 2025

@author: georgeestrella
"""

# Recontruccion de Series Temporales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import yfinance as yf


start_date = '2010-01-01'
end_date = '2024-12-12'
ticker = ['SQM-B.SN', 'VAPORES.SN']
mpor = 1

# Series Temporales
data = yf.download(ticker, start=start_date, end=end_date)['Close']

# Vector Binario
random.seed(50)
binario = pd.Series([random.choice([0, 1]) for _ in range(len(data))], index=data.index)

for col in data.columns:
    binary_price = binario*data[col]
    data[col+'_Binary'] = binary_price
    data.drop(col, axis = 1, inplace=True)
    data[col+'_Linear'] = binary_price.interpolate(method='linear')
    data[col+'_Polynomial'] = binary_price.interpolate(method='polynomial', order=3)
    data[col+'_Spline'] = binary_price.interpolate(method='spline', order=3)
    data[col+'_Akima'] = binary_price.interpolate(method='akima')
    data[col+'_Nearest'] = binary_price.interpolate(method='nearest')
    data[col+'_PCHIP_Price'] = binary_price.interpolate(method='pchip')
    data[col+'_BackwardFill'] = binary_price.bfill()
    data[col+'_FordwardFill'] = binary_price.ffill()


data = data.melt(id_vars='Date', var_name='Nemo', value_name='Precio')















