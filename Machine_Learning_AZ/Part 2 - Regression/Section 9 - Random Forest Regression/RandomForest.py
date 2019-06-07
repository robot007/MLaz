# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:56:28 2018

@author: zhensong
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
X, y = make_regression(n_features=1, n_informative=2,random_state=0, shuffle=False)
plt.subplot(2,1,1)
plt.plot(X)
plt.subplot(2,1,2)
plt.plot(y)
plt.show()

regr = RandomForestRegressor(max_depth=4, random_state=10)
regr.fit(X,y)
print(regr.feature_importances_)
print(regr.predict([[0]]))

base = np.linspace(0,100, num=100)
X_grid=np.arange(min(X), max(X), 0.1)
y_pred = regr.predict(X)
plt.plot(X, y_pred, 'r')
plt.scatter(X, y)
plt.show()