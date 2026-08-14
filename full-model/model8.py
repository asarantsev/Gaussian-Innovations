import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy

# This is the file where we have 5 asset classes:
# 3 geometric stock returns: S&P, developed, emerging
# 2 arithmetic bond returns: Corporate investment-grade and Treasury
# There are 4 factors: stock valuation measure and volatility
# BAA and Treasury bond rates
# All data is annual and nominal, not inflation-adjusted

# reading the data file
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['BAA'].values
long = DF['Treasury'].values
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

# plot of volatility
plt.plot(range(1928, 1928 + N), vol)
plt.title('Volatility')
plt.savefig('vol.png')
plt.close()

# log plot of S&P level
plt.plot(range(1927, 1928 + N), price)
plt.title('Index')
plt.yscale('log')
plt.savefig('index.png')
plt.close()

# graph of the two bond rates
plt.plot(range(1927, 1928 + N), rates, label = 'BAA')
plt.plot(range(1927, 1928 + N), long, label = '10-year Treasury')
plt.title('Bond Rates')
plt.savefig('rates.png')
plt.close()

# log scale plot of annual dividends
plt.plot(range(1927, 1928 + N), div)
plt.title('Dividends')
plt.yscale('log')
plt.savefig('div.png')
plt.close()

# total returns
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))
premeasure = np.log(wealth/div) # measure before detrending

# regression equation for computation of the valuation measure
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N + 1)

# Fitting autoregression for the valuation measure with stochastic volatility
RegMeasure = OLS(np.diff(measure)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()
print('simple autoregression for the valuation measure with stochastic volatility')
print(RegMeasure.summary())

# current and long-term average 
print('average measure = ', np.mean(measure))
print('end of 2025 measure = ', measure[-1])

# graph of the new valuation measure
plt.plot(range(1927, 1928 + N), measure)
plt.title('Measure')
plt.savefig('measure.png')
plt.close()

# Fitting various regressions for the USA normalized geometric stock returns
# Here is the spread: BAA - long, the version which we finally choose
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'spread' : (rates - long)[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, mainDF).fit()
print('Regression for normalized geometric returns of the S&P')
print('versus duration, valuation, spread')
print(RegUSA.summary())

# Regression for geometric normalized returns of international developed stocks
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()
print('Regression for normalized geometric returns of developed markets')
print('with duration, spreads, and the valuation measure')
print(RegIntl.summary())

# Regression for geometric normalized returns of international emerging stocks
nEMRet = np.log(np.ones(38) + em)/vol[60:]
RegEM = OLS(nEMRet, mainDF.iloc[60:]).fit()
print('Regression for normalized geometric returns of emerging markets')
print('with duration, spreads, and the valuation measure')
print(RegEM.summary())

# Autoregression of corporate bond rates with stochastic volatility
print('Autoregression of corporate bond rates with stochastic volatility')
RegBondRates = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()
print(RegBondRates.summary())

# Autoregression of annual volatility on log scale
print('Autoregression of log volatility')
RegVol = OLS(np.diff(np.log(vol)), pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
print(RegVol.summary())

# Corporate bond returns vs rates regression
print('Arithmetic corporate bond returns regression with stochastic volatility')
RegBondReturns = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
print(RegBondReturns.summary())

# Autoregression of risk log spread of logs
print('Log risk spread of log rates')
spreads = np.log(rates) - np.log(long)
lspreads = np.log(spreads)
print('Autoregression of order 1')
RegRiskSpreads = OLS(np.diff(lspreads), pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()
print(RegRiskSpreads.summary())

# print the covariace and correlation matrix for residuals, 
# there are eight series since 4 + 5 = 9 equations
# but the equation for Treasury bond returns is deterministic
allResid = [RegUSA.resid, RegIntl.resid, RegEM.resid, RegBondReturns.resid, RegVol.resid, RegBondRates.resid, RegMeasure.resid, RegRiskSpreads.resid]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl-full', 'em-full', 'bondReturns', 'vol', 'bondRates', 'measure', 'spreads']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(8):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()

# we print it column by column since the screen does not fit
# the entire 8x8 matrices
for name in allNames:
    print(covMatrix[name]*10000)
    print(corrMatrix[name])