import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    print(acf(data, qstat = True, nlags = 10)[2])
    print(acf(abs(data), qstat = True, nlags = 10)[2])
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    print(label)
    return np.std(data)
    
DF = pd.read_excel('data2025.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['Rates'].values
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
N = 98

plt.plot(range(1928, 1928 + N), vol)
plt.title('Volatility')
plt.savefig('vol.png')
plt.close()

plt.plot(range(1927, 1928 + N), price)
plt.title('Index')
plt.savefig('index.png')
plt.yscale('log')
plt.close()

plt.plot(range(1927, 1928 + N), rates)
plt.title('BAA Rate')
plt.savefig('baa.png')
plt.close()

plt.plot(range(1927, 1928 + N), div)
plt.title('Dividends')
plt.savefig('div.png')
plt.yscale('log')
plt.close()

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))

plt.plot(range(1927, 1928 + N), wealth)
plt.title('Wealth')
plt.savefig('wealth.png')
plt.yscale('log')
plt.close()

premeasure = np.log(wealth/div)
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N + 1)
resMeas = measureReg.resid
plt.plot(range(1927, 1928 + N), measure)
plt.title('Measure')
plt.savefig('measure.png')
plt.close()

mReg = OLS(np.diff(measure), pd.DataFrame({'const' : 1, 'lag' : measure[:-1]})).fit()
print(mReg.params)

nUSAret = total/vol
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
normBonds = np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:]

RegUSA = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})).fit()
print('Regression for the USA returns')
print(RegUSA.summary())
resUSA = RegUSA.resid
print(RegUSA.params)

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
print('Regression for the log volatility')
print(RegVol.summary())
resVol = RegVol.resid
print(RegVol.params)

nRates = np.diff(np.log(rates))/vol
logRates = np.log(rates)
RegRates = OLS(np.diff(logRates)/vol, pd.DataFrame({'vol' : 1/vol, 'lag' : logRates[:-1]/vol})).fit()
print('Regression for Rates')
print(RegRates.summary())
resRates = RegRates.resid

print('Regression for international returns')
RegIntl = OLS(nIntlRet, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})[42:]).fit()
print(RegIntl.summary())
resIntl = RegIntl.resid

print('bond returns')
RegBonds = OLS(normBonds, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1}).iloc[45:]).fit()
print(RegBonds.summary())
resBonds = RegBonds.resid

allResid = [resUSA, resIntl, resVol, resRates, resBonds, resMeas]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'vol', 'rates', 'bonds', 'measure']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(6):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    print(plots(allResid[k], allNames[k])) # normality and autocorrelation function plots
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(covMatrix*10000)
print(corrMatrix)