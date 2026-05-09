import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats 

def plots(data, label):
#     plot_acf(data, zero = False)
#     plt.title(label + '\n ACF for Original Values')
#     plt.savefig('O-' + label + '.png')
#     plt.close()
#     plot_acf(abs(data), zero = False)
#     plt.title(label + '\n ACF for Absolute Values')
#     plt.savefig('A-' + label + '.png')
#     plt.close()
#     qqplot(data, line = 's')
#     plt.title(label + '\n Quantile-Quantile Plot vs Normal')
#     plt.savefig('QQ-' + label + '.png')
#     plt.close()
    print(label)
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    # return np.std(data)
    
DF = pd.read_excel('data2025-new.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['Rates'].values
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
N = 98

# plt.plot(range(1928, 1928 + N), vol)
# plt.title('Volatility')
# plt.savefig('vol.png')
# plt.close()
# 
# plt.plot(range(1927, 1928 + N), price)
# plt.title('Index')
# plt.savefig('index.png')
# plt.close()
# 
# plt.plot(range(1927, 1928 + N), rates)
# plt.title('BAA Rate')
# plt.savefig('baa.png')
# plt.close()
# 
# plt.plot(range(1927, 1928 + N), div)
# plt.title('Dividends')
# plt.savefig('div.png')
# plt.close()

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
normBonds = np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:]

RegUSA = OLS(nUSAret, mainDF).fit()
print('Regression for the USA returns')
print(RegUSA.summary())
resUSA = RegUSA.resid

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
print('Regression for the log volatility')
print(RegVol.summary())
resVol = RegVol.resid

RegRates = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol, 'vol' : 1})).fit()
print('Regression for Rates')
print(RegRates.summary())
resRates = RegRates.resid
plots(resRates, 'rates')

print('Regression for international returns')
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()
print(RegIntl.summary())
resIntl = RegIntl.resid

print('bond returns')
RegBonds = OLS(normBonds, mainDF[['const', 'duration']].iloc[45:]).fit()
print(RegBonds.summary())
resBonds = RegBonds.resid

allResid = [resUSA, resIntl, resVol, resRates, resBonds]
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'vol', 'rates', 'bonds']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(5):
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    print(plots(allResid[k], allNames[k])) # normality and autocorrelation function plots
    
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(covMatrix*10000)
print(corrMatrix)