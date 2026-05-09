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
    
DF = pd.read_excel('data2025-new.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['Rates'].values
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
spread = DF['Spreads'].values
N = 98

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

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'vol' : 1})
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
normBonds = np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:]

RegSpread = OLS(np.diff(spread), pd.DataFrame({'const' : 1, 'lag' : spread[:-1]})).fit()
print(RegSpread.summary())
resSpread = RegSpread.resid
plots(resSpread, 'spread')

RegVolSpread = OLS(np.diff(spread)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spread[:-1]/vol, 'vol' : 1})).fit()
print(RegVolSpread.summary())
resVolSpread = RegVolSpread.resid
plots(resVolSpread, 'vol-spread')

RegVolSpread = OLS(np.diff(spread)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spread[:-1]/vol})).fit()
print(RegVolSpread.summary())
resVolSpread = RegVolSpread.resid
plots(resVolSpread, 'vol-spread')

RegVolSpread = OLS(np.diff(spread)/vol, pd.DataFrame({'vol' : 1, 'lag' : spread[:-1]/vol})).fit()
print(RegVolSpread.summary())
resVolSpread = RegVolSpread.resid
plots(resVolSpread, 'vol-spread')

RegVolSpread = OLS(np.diff(spread)/vol, pd.DataFrame({'lag' : spread[:-1]/vol})).fit()
print(RegVolSpread.summary())
resVolSpread = RegVolSpread.resid
plots(resVolSpread, 'vol-spread')




