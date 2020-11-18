# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:13:03 2020

@author: HEENA KAUSAR
"""

#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#READING THE DATA IN THE FILE
data = pd.read_csv('advertising.csv')
data.head()


#DATA EXPLORATION
fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind ='scatter', x='TV',y='Sales',ax=axs[0], figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#CREATING X&Y FOR LINEAR REGRESSION
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales


#IMPORTING LINEAR REGRESSION ALGORITHM
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



result = 12.23 + 0.1244 * 50
print(result)


#CREATE DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='0.55',Linewidth=3)


import statsmodels.formula.api as smf
lr = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lr.conf_int()

#FINDING THE PROBABILITY VALUE
lr.pvalues


#FINDING THE R-SQUARE VALUES
lr.rsquared

 
#MULTI LINEAR REGRESSION
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lr = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
lr.conf_int()
lr.summary()