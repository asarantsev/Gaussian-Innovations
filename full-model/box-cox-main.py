import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats

def verification(data, label):
    print(label)
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)
   
def BoxCox(data, label):
    BC = scipy.stats.boxcox(data)
    print(label)
    print('order = ', BC[1])
    new = BC[0]
    return new
    
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
price = DF['Price'].values
N = len(price) - 1
div = DF['Dividends'].values
dev = DF['International'].values[43:]
em = DF['Emerging'].values[61:]
vol = DF['Volatility'].values[1:]
usRet = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
devRet = np.log(np.ones(56) + dev)
emRet = np.log(np.ones(38) + em)
nUSRet = usRet/vol
nDevRet = devRet/vol[42:]
nEmRet = emRet/vol[60:]

verification(usRet, 'usa')
verification(nUSRet, 'norm-usa')
verification(BoxCox(np.exp(usRet), 'BCX-usa'), 'BCX-usa')
verification(BoxCox(np.exp(nUSRet), 'BCX-norm-usa'), 'BCX-norm-usa')

verification(devRet, 'dev')
verification(nDevRet, 'norm-dev')
verification(BoxCox(np.exp(devRet), 'BCX-dev'), 'BCX-dev')
verification(BoxCox(np.exp(nDevRet), 'BCX-norm-dev'), 'BCX-norm-dev')

verification(emRet, 'em')
verification(nEmRet, 'norm-em')
verification(BoxCox(np.exp(emRet), 'BCX-em'), 'BCX-em')
verification(BoxCox(np.exp(nEmRet), 'BCX-norm-em'), 'BCX-norm-em')