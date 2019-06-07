# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:42:24 2019

@author: zhensong
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# use R formula
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head()
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
mod2 = smf.ols(formula='Lottery ~ Literacy : Wealth', data=df)
res = mod.fit()
res2 = mod2.fit()
print(res.summary())
print(res2.summary())

y=df.Literacy*(-0.378)+0.4138*df.Wealth+39.7979

