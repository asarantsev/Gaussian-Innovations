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

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]

print('USA returns')
plots(nUSAret, 'simplest USA')
print('model with duration')
RegU1 = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol})).fit()
plots(RegU1.resid, 'duration USA')
RegU2 = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()
plots(RegU2.resid, 'vol USA')
RegU3 = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})).fit()
plots(RegU3.resid, 'vol and duration USA')

print('international returns')
plots(nIntlRet, 'simplest')
print('model with duration')
RegI1 = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'duration' : -np.diff(rates)[42:]/vol[42:]})).fit()
plots(RegI1.resid, 'duration')
RegI2 = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'vol' : 1})).fit()
plots(RegI2.resid, 'vol')
RegI3 = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol[42:], 'duration' : -np.diff(rates)[42:]/vol[42:], 'vol' : 1})).fit()
plots(RegI3.resid, 'vol and duration')

print('bond returns')
RegL0 = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
plots(RegL0.resid, 'gometric returns, no vol')
RegL1 = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1}).iloc[45:]).fit()
plots(RegL1.resid, 'geometric returns, full')
RegL2 = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
plots(RegL2.resid, 'geometric returns, simple')
RegA0 = OLS((bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1] - np.ones(len(bonds)-1))/vol[45:], pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
plots(RegA0.resid, 'arithmetic returns, no vol')
RegA1 = OLS((bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1] - np.ones(len(bonds)-1))/vol[45:], pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1}).iloc[45:]).fit()
plots(RegA1.resid, 'arithmetic returns, vol')
RegA2 = OLS((bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1] - np.ones(len(bonds)-1))/vol[45:], pd.DataFrame({'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()
plots(RegA2.resid, 'arithmetic returns, simple')

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
print('Regression for the log volatility')
plots(RegVol.resid, 'vol')

print('Regression for Rates')
RegR0 = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'vol' : 1, 'lag' : np.log(rates)[:-1]/vol})).fit()
plots(RegR0.resid, 'no const')
RegR1 = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()
plots(RegR1.resid, 'no vol')
RegR2 = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol, 'vol' : 1})).fit()
plots(RegR2.resid, 'full')
print('without logarithms')
RegB0 = OLS(np.diff(rates)/vol, pd.DataFrame({'vol' : 1, 'lag' : rates[:-1]/vol})).fit()
plots(RegB0.resid, 'no const')
RegB1 = OLS(np.diff(rates)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : rates[:-1]/vol})).fit()
plots(RegB1.resid, 'no vol')
RegB2 = OLS(np.diff(rates)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : rates[:-1]/vol, 'vol' : 1})).fit()
plots(RegB2.resid, 'full')

print('RW for Rates')
plots(np.diff(rates), 'arithmetic-RW')
plots(np.diff(rates)/vol, 'arithmetic-RW-SV')
plots(np.diff(np.log(rates)), 'geometric-RW')
plots(np.diff(np.log(rates))/vol, 'geometric-RW-SV')
print('with additive volatility')
RegD1 = OLS(np.diff(rates)/vol, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()
plots(RegD1.resid, 'arithmetic-RW')
RegD2 = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()
plots(RegD2.resid, 'geometric-RW')

print('Chosen model for domestic stocks')
print(RegU3.summary())
print('Chosen model for international stocks')
print(RegI3.summary())
print('Chosen model for domestic bonds')
print(RegL2.summary())
print('Chosen model for stock volatility')
print(RegVol.summary())
print('Chosen model for bond rates')
print(RegR1.summary())

allResid = [RegU3.resid, RegI3.resid, RegL2.resid, RegVol.resid, RegR1.resid]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'bonds', 'vol', 'rates']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(5):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(covMatrix*10000)
print(corrMatrix)
