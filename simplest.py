import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from verification import plots
    
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

plt.plot(range(1928, 1928 + N), vol)
plt.title('Volatility')
plt.savefig('vol.png')
plt.close()

plt.plot(range(1927, 1928 + N), price)
plt.title('Index')
plt.savefig('index.png')
plt.close()

plt.plot(range(1927, 1928 + N), rates)
plt.title('BAA Rate')
plt.savefig('baa.png')
plt.close()

plt.plot(range(1927, 1928 + N), div)
plt.title('Dividends')
plt.savefig('div.png')
plt.close()

def center(data):
    return data - np.mean(data) * np.ones(len(data))

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
normBonds = np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:]

print('USA returns')
print('mean = ', np.mean(nUSAret))
resUSA = center(nUSAret)

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
print('Regression for the log volatility')
print(RegVol.summary())
resVol = RegVol.resid

regRates = OLS(np.diff(rates)/vol, pd.DataFrame({'const' : 1/vol, 'slope' : rates[:-1]/vol, 'vol' : 1})).fit()
print('Regression for the log rates')
print(regRates.summary())
resRates = regRates.resid

print('International returns')
print('mean = ', np.mean(nIntlRet))
resIntl = center(nIntlRet)

print('bond returns')
RegBonds = OLS(normBonds, mainDF.iloc[45:]).fit()
print(RegBonds.summary())
resBonds = RegBonds.resid

allResid = [resUSA, resIntl, resBonds, resVol, resRates]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'bonds', 'vol', 'rates']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(5):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    print(plots(allResid[k], allNames[k])) # normality and autocorrelation function plots
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()

print(covMatrix*10000)
print(corrMatrix)
