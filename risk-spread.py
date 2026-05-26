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
spreads = baa - long
lspreads = np.log(spreads)
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
ntotal = total/vol

RegSpreads = OLS(spreads[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spreads[:-1]/vol, 'vol' : 1})).fit()
print(RegSpreads.summary())
plots(RegSpreads.resid, 'vol-spreads')
Reg1 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : spreads[:-1]/vol})).fit()
plots(Reg1.resid, 'returns1')
Reg2 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : spreads[:-1]})).fit()
plots(Reg2.resid, 'returns2')
Reg3 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : spreads[:-1]})).fit()
plots(Reg3.resid, 'returns3')
Reg4 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : spreads[:-1]})).fit()
plots(Reg4.resid, 'returns4')
Reg5 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : spreads[:-1]/vol})).fit()
plots(Reg5.resid, 'returns5')
Reg6 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : spreads[:-1]/vol})).fit()
plots(Reg6.resid, 'returns6')
RegL1 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : lspreads[:-1]/vol})).fit()
plots(RegL1.resid, 'returns-log1')
RegL2 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'spreads' : lspreads[:-1]})).fit()
plots(RegL2.resid, 'returns-log2')
RegL3 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : lspreads[:-1]})).fit()
plots(RegL3.resid, 'returns-log3')
RegL4 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : lspreads[:-1]})).fit()
plots(RegL4.resid, 'returns-log4')
RegL5 = OLS(ntotal, pd.DataFrame({'const' : 1/vol, 'spreads' : lspreads[:-1]/vol})).fit()
plots(RegL5.resid, 'returns-log5')
RegL6 = OLS(ntotal, pd.DataFrame({'const' : 1, 'spreads' : lspreads[:-1]/vol})).fit()
plots(RegL6.resid, 'returns-log6')