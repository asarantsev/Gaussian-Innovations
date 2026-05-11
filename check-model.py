import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats 

def plots(data, label):
    print(label)
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    
DF = pd.read_excel('data2025.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['Rates'].values
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
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
RegM = OLS(measure[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()
print('simple autoregression for the valuation measure')
print(RegM.summary())

print('average measure = ', np.mean(measure))
print('end of 2025 measure = ', measure[-1])

plots(RegM.resid, 'measure-vol')

mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, mainDF).fit()
print('Regression for domestic returns')
print(RegUSA.summary())
plots(RegUSA.resid, 'usa')

nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()
print('Regression for international returns')
print(RegIntl.summary())
plots(RegIntl.resid, 'intl')

RegI3 = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'duration' : -np.diff(rates)[42:]/vol[42:], 'vol' : 1})).fit()
print(RegI3.summary())
plots(RegI3.resid, 'intl')

RegR1 = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()
RegVol = OLS(np.log(vol)[1:], pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
RegL2 = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()

allResid = [RegUSA.resid, RegI3.resid, RegL2.resid, RegVol.resid, RegR1.resid, RegM.resid]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'bonds', 'vol', 'rates', 'measure']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(6):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(covMatrix*10000)
print(corrMatrix)