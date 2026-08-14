# run check-full-model.py
from verification import plots
import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats

DF0 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread' : (rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF0).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, spread')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-spreads')

DF1 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'log-spread' : np.log(rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF1).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, log spread')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-log-spreads')

DF2 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread-log' : (np.log(rates) - np.log(long))[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF2).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, spread of logs')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-spread-log')

DF3 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'log-spread-log' : np.log(np.log(rates) - np.log(long))[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF3).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, log spread of logs')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-log-spread-log')