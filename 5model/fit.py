import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.api import stats

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)
    
def invM(factor):
    return np.linalg.inv(np.array([[len(factor), sum(factor)], [sum(factor), sum(factor**2)]]))

# reading the data file
DF = pd.read_excel('data2025.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['Rates'].values
BONDLAG = 45
INTLLAG = 43
bonds = DF['Bonds'].values[BONDLAG:]
intl = DF['International'].values[INTLLAG:]
N = 98 # overall number of data points

# total returns
stockRet = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
DFstocks = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol})
RegStocks = OLS(stockRet/vol, DFstocks).fit()

# apply Yeo-Johnson transform to log volatility and fit autoregression for transformed volatility
lvol = np.log(vol)
tvol = scipy.stats.yeojohnson(lvol, -0.5)
DFvol = pd.DataFrame({'const' : 1, 'lag' : tvol[:-1]})
RegVol = OLS(np.diff(tvol), DFvol).fit()

# Arithmetic corporate bond returns vs rates regression with stochastic volatility
bondRet = np.diff(np.log(bonds))
DFbonds = pd.DataFrame({'const' : 1, 'duration' : -np.diff(rates)[BONDLAG:]})
RegBonds = OLS((bondRet - 0.01 * rates[BONDLAG:-1]), DFbonds).fit()

# Log rates as an autoregression with stochastic volatility
trates = scipy.stats.yeojohnson(rates, -0.5)
DFrates = pd.DataFrame({'const' : 1, 'lag' : trates[:-1]})
RegRates = OLS(np.diff(trates), DFrates).fit()

# International stocks
intlRet = np.log(1 + intl)
DFintl = DFstocks.iloc[INTLLAG-1:]
RegIntl = OLS(intlRet/vol[INTLLAG-1:], DFintl).fit()

allRegressions = [RegVol, RegRates, RegStocks, RegBonds, RegIntl]
allDFs = [DFvol, DFrates, DFstocks, DFbonds, DFintl]
allResid = [Reg.resid for Reg in allRegressions]
lengths = [len(res) for res in allResid]
allNames = ['vol', 'rate', 'stocks', 'bonds', 'intl']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(5):
    print('residuals for')
    print(allNames[k])
    print(allRegressions[k].summary())
    print(allRegressions[k].params)
    verification(allResid[k])
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    print(np.dot(np.transpose(allDFs[k].values), allDFs[k].values))

covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()
print(np.round(covMatrix*(10**4), 2))
print(corrMatrix)

