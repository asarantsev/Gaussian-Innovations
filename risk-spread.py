import numpy as np
import pandas as pd
from statsmodels.api import OLS
from verification import plots

DF = pd.read_excel('data.xlsx', sheet_name = 'data')
baa = DF['BAA'].values
long = DF['Treasury'].values
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
N = 98
# Risk spreads, but edit to
# spreads = np.log(baa) - np.log(long) 
# to check another case
spreads = baa - long
lspreads = np.log(spreads)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
ntotal = total/vol

# First, for complete picture, autoregression of spreads with lag 1 with stochastic volatility
# It was really in another piece of code, but we still include this here
RegSpreads = OLS(spreads[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spreads[:-1]/vol, 'vol' : 1})).fit()
print(RegSpreads.summary())
plots(RegSpreads.resid, 'vol-spreads')

# Now many regressions for normalized S&P returns versus spreads
# in different combinations
Reg0 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : spreads[:-1]/vol})).fit()
Reg1 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : spreads[:-1]})).fit()
Reg2 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : spreads[:-1]})).fit()
Reg3 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : spreads[:-1]})).fit()
Reg4 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : spreads[:-1]/vol})).fit()
Reg5 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : spreads[:-1]/vol})).fit()

# Regressions for normalized returns versus log spreads
# in different combinations
RegL0 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : lspreads[:-1]/vol})).fit()
RegL1 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : lspreads[:-1]})).fit()
RegL2 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : lspreads[:-1]})).fit()
RegL3 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : lspreads[:-1]})).fit()
RegL4 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : lspreads[:-1]/vol})).fit()
RegL5 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : lspreads[:-1]/vol})).fit()

# labels
labels = ['returns-vs-spreads-' + str(k) for k in range(6)] + ['returns-vs-logspreads-' + str(k) for k in range(6)]
# Now print the output and the analysis of residuals
for Reg in [Reg0, Reg1, Reg2, Reg3, Reg4, Reg5, RegL0, RegL1, RegL2, RegL3, RegL4, RegL5]:
  print(Reg.summary())
  plots(Reg.resid, 'returns-log6')
