import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats
from verification import plots

# reading the data file
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['BAA'].values
long = DF['Treasury'].values

# the last three entries have less than full data points
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
em = DF['Emerging'].values[61:]
N = 98 # overall number of data points

# possible initial conditions for simulations: long-term averages or 2025 values
print('Average volatility = ', np.mean(vol))
print('Average BAA rate = ', np.mean(rates))
print('Average long rate = ', np.mean(long))
print('2025 volatility = ', vol[-1])
print('End of 2025 BAA rate = ', rates[-1])
print('End of 2025 long rate = ', long[-1])

# total returns
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))
premeasure = np.log(wealth/div) # measure before detrending

# regression equation for computation of the valuation measure
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N + 1)
resMeas = measureReg.resid
plots(resMeas, 'measure') # testing residuals, they do not pass IID Gaussian

# Fitting autoregression for the valuation measure with stochastic volatility
RegMeasure = OLS(measure[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()
print('simple autoregression for the valuation measure')
print(RegMeasure.summary())

print('average measure = ', np.mean(measure))
print('end of 2025 measure = ', measure[-1])
plots(RegMeasure.resid, 'measure-vol')

# Fitting various regressions for the USA normalized geometric stock returns
# This version is without spreads
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, mainDF).fit()
print('Regression for normalized geometric returns of the S&P')
print('with duration, spreads, and the valuation measure')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa')

# And now four various versions with risk spreads. 
# Here is the spread: BAA - long, the version which we finally choose
DF0 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread' : (rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF0).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, spread')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-spreads')

# Next is the log spread of rates
DF1 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'log-spread' : np.log(rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF1).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, log spread')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-log-spreads')

# This is the spread of log rates
DF2 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread-log' : (np.log(rates) - np.log(long))[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF2).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, spread of logs')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-spread-log')

# And finally the log spread of log rates
DF3 = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'log-spread-log' : np.log(np.log(rates) - np.log(long))[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, DF3).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, log spread of logs')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa-log-spread-log')

# Regression for geometric normalized returns of international developed stocks
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()
print('Regression for normalized geometric returns of developed markets')
print('with duration, spreads, and the valuation measure')
print(RegIntl.summary())
plots(RegIntl.resid, 'intl-full')

# Compare it with reduced regression, with valuation measure and risk spreads removed
# We choose the full regression
print('Cut regression for normalized geometric returns of developed markets')
print('with duration but without spreads and the valuation measure')
RegIntlcut = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'duration' : -np.diff(rates)[42:]/vol[42:], 'vol' : 1})).fit()
print(RegIntlcut.summary())
plots(RegIntlcut.resid, 'intl-cut')

# Regression for geometric normalized returns of international emerging stocks
nEMRet = np.log(np.ones(38) + em)/vol[60:]
RegEM = OLS(nEMRet, mainDF.iloc[60:]).fit()
print('Regression for normalized geometric returns of emerging markets')
print('with duration, spreads, and the valuation measure')
print(RegEM.summary())
plots(RegEM.resid, 'em-full')

# Compare it with reduced regression, with valuation measure and risk spreads removed
# We choose the full regression
print('Cut regression for normalized geometric returns of emerging markets')
print('with duration but without the valuation measure')
RegEMcut = OLS(nEMRet, pd.DataFrame({'const' : 1/vol[60:], 'duration' : -np.diff(rates)[60:]/vol[60:], 'vol' : 1})).fit()
print(RegEMcut.summary())
plots(RegEMcut.resid, 'em-cut')

# Autoregression of corporate bond rates with stochastic volatility
print('Autoregression of corporate bond rates with stochastic volatility')
RegBondRates = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()
print(RegBondRates.summary())
plots(RegBondRates.resid, 'bondRates')

# Autoregression of annual volatility on log scale
print('Autoregression of log volatility')
RegVol = OLS(np.log(vol)[1:], pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
print(RegVol.summary())
plots(RegVol.resid, 'vol')

# Corporate bond returns vs rates regression
print('Arithmetic corporate bond returns regression with stochastic volatility')
RegBondReturns = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
print(RegBondReturns.summary())
plots(RegBondReturns.resid, 'bondReturns')

# Autoregression of risk log spread of logs
print('Log risk spread of log rates')
spreads = np.log(rates) - np.log(long)
lspreads = np.log(spreads)
print('Autoregression of order 1')

RegRiskSpreads = OLS(np.diff(lspreads), pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()
print(RegRiskSpreads.summary())
plots(RegRiskSpreads.resid, 'spreads')

# print the covariace and correlation matrix for residuals, 
# there are eight series 
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
