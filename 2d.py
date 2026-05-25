import numpy as np
import pandas as pd
from statsmodels.api import OLS
from verification import plots

DF = pd.read_excel('data.xlsx', sheet_name = 'data')
baa = DF['BAA'].values
long = DF['Treasury'].values
vol = DF['Volatility'].values[1:]

Reg1 = OLS(long[1:], pd.DataFrame({'const' : 1, 'lag' : long[:-1]})).fit()
print(Reg1.summary())
plots(Reg1.resid, 'only-long')

Reg2 = OLS(long[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : long[:-1]/vol, 'vol' : 1})).fit()
print(Reg2.summary())
plots(Reg2.resid, 'vol-long')

RegBAA = OLS(baa[1:], pd.DataFrame({'const' : 1, 'baa' : baa[:-1], 'long' : long[:-1]})).fit()
print(RegBAA.summary())
plots(RegBAA.resid, 'baa')
RegLong = OLS(long[1:], pd.DataFrame({'const' : 1, 'baa' : baa[:-1], 'long' : long[:-1]})).fit()
print(RegLong.summary())
plots(RegLong.resid, 'long')

DFReg = pd.DataFrame({'const' : 1/vol, 'baa' : baa[:-1]/vol, 'long' : long[:-1]/vol, 'vol' : 1})
RegBAA = OLS(baa[1:]/vol, DFReg).fit()
print(RegBAA.summary())
plots(RegBAA.resid, 'baa')
RegLong = OLS(long[1:]/vol, DFReg).fit()
print(RegLong.summary())
plots(RegLong.resid, 'long')

spreads = baa - long
Reg = OLS(spreads[1:], pd.DataFrame({'const' : 1, 'lag' : spreads[:-1]})).fit()
print(Reg.summary())
plots(Reg.resid, 'spreads')
RegVol = OLS(spreads[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spreads[:-1]/vol, 'vol' : 1})).fit()
print(RegVol.summary())
plots(RegVol.resid, 'vol-spreads')