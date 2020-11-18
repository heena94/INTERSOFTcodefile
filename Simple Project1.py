# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:53:51 2020

@author: HEENA KAUSAR
"""

#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#READING THE DATA FROM YOUR FILES
data = pd.read_csv('EUR_USD Historical Data.csv',usecols=[0,1,2,3,4])

content_avg = data[['Price','Open','High','Low']].mean(axis=1)

aa = np.arange(1,len(data)+1,1)

plt.plot(aa,content_avg,'0.23',label='MY FIRST PLOT')
plt.savefig('figure.png')

#

