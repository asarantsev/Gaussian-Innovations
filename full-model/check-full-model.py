import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats
from verification import plots

DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['BAA'].values
bonds = DF['Bonds'].values[45:]
long = DF['Treasury'].values
intl = DF['International'].values[43:]
em = DF['Emerging'].values[61:]
N = 98

print('Average volatility = ', np.mean(vol))
print('Average rate = ', np.mean(rates))
print('2025 volatility = ', vol[-1])
print('End of 2025 rate = ', rates[-1])

total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))
premeasure = np.log(wealth/div)
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N + 1)
resMeas = measureReg.resid
plots(resMeas, 'measure')
RegMeasure = OLS(measure[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()
print('simple autoregression for the valuation measure')
print(RegMeasure.summary())

print('average measure = ', np.mean(measure))
print('end of 2025 measure = ', measure[-1])
plots(RegMeasure.resid, 'measure-vol')

mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread' : (rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, mainDF).fit()
print('Regression for normalized geometric returns of the S&P')
print('with duration, spreads, and the valuation measure')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa')

nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()
print('Regression for normalized geometric returns of developed markets')
print('with duration, spreads, and the valuation measure')
print(RegIntl.summary())
plots(RegIntl.resid, 'intl-full')

print('Cut regression for normalized geometric returns of developed markets')
print('with duration but without spreads and the valuation measure')
RegIntlcut = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'duration' : -np.diff(rates)[42:]/vol[42:], 'vol' : 1})).fit()
print(RegIntlcut.summary())
plots(RegIntlcut.resid, 'intl-cut')

nEMRet = np.log(np.ones(38) + em)/vol[60:]
RegEM = OLS(nEMRet, mainDF.iloc[60:]).fit()
print('Regression for normalized geometric returns of emerging markets')
print('with duration, spreads, and the valuation measure')
print(RegEM.summary())
plots(RegEM.resid, 'em-full')

print('Cut regression for normalized geometric returns of emerging markets')
print('with duration but without the valuation measure')
RegEMcut = OLS(nEMRet, pd.DataFrame({'const' : 1/vol[60:], 'duration' : -np.diff(rates)[60:]/vol[60:], 'vol' : 1})).fit()
print(RegEMcut.summary())
plots(RegEMcut.resid, 'em-cut')

print('Autoregression of corporate bond rates with stochastic volatility')
RegBondRates = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()
print(RegBondRates.summary())
plots(RegBondRates.resid, 'bondRates')

print('Autoregression of log volatility')
RegVol = OLS(np.log(vol)[1:], pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
print(RegVol.summary())
plots(RegVol.resid, 'vol')

print('Arithmetic corporate bond returns regression with stochastic volatility')
RegBondReturns = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
print(RegBondReturns.summary())
plots(RegBondReturns.resid, 'bondReturns')

print('Log risk spread of log rates')
spreads = np.log(rates) - np.log(long)
lspreads = np.log(spreads)
print('Autoregression of order 1')
RegRiskSpreads = OLS(np.diff(lspreads), pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()
print(RegRiskSpreads.summary())
plots(RegRiskSpreads.resid, 'spreads')

allResid = [RegUSA.resid, RegIntl.resid, RegEM.resid, RegBondReturns.resid, RegVol.resid, RegBondRates.resid, RegMeasure.resid, RegRiskSpreads.resid]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl-full', 'em-full', 'bondReturns', 'vol', 'bondRates', 'measure', 'spreads']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(8):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(covMatrix*10000)
print(corrMatrix)