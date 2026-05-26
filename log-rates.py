import numpy as np
import pandas as pd
from statsmodels.api import OLS
from verification import plots

DF = pd.read_excel('data.xlsx', sheet_name = 'data')
baa = np.log(DF['BAA'].values)
long = np.log(DF['Treasury'].values)
vol = DF['Volatility'].values[1:]

Reg1 = OLS(long[1:], pd.DataFrame({'const' : 1, 'lag' : long[:-1]})).fit()
print(Reg1.summary())
plots(Reg1.resid, 'long1')

Reg2 = OLS(long[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : long[:-1]/vol, 'vol' : 1})).fit()
print(Reg2.summary())
plots(Reg2.resid, 'long2')

Reg3 = OLS(long[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : long[:-1]/vol})).fit()
print(Reg3.summary())
plots(Reg3.resid, 'long3')

Reg4 = OLS(long[1:]/vol, pd.DataFrame({'vol' : 1, 'lag' : long[:-1]/vol})).fit()
print(Reg4.summary())
plots(Reg4.resid, 'long4')

RegBAA = OLS(baa[1:], pd.DataFrame({'const' : 1, 'baa' : baa[:-1], 'long' : long[:-1]})).fit()
print(RegBAA.summary())
plots(RegBAA.resid, 'baa')
RegLong = OLS(long[1:], pd.DataFrame({'const' : 1, 'baa' : baa[:-1], 'long' : long[:-1]})).fit()
print(RegLong.summary())
plots(RegLong.resid, 'long')

DFCut = pd.DataFrame({'const' : 1/vol, 'baa' : baa[:-1]/vol, 'long' : long[:-1]/vol})
RegBAA = OLS(baa[1:]/vol, DFCut).fit()
print(RegBAA.summary())
plots(RegBAA.resid, 'baa')
RegLong = OLS(long[1:]/vol, DFCut).fit()
print(RegLong.summary())
plots(RegLong.resid, 'long')

DFNew = pd.DataFrame({'vol' : 1, 'baa' : baa[:-1]/vol, 'long' : long[:-1]/vol})
RegBAA = OLS(baa[1:]/vol, DFNew).fit()
print(RegBAA.summary())
plots(RegBAA.resid, 'baa')
RegLong = OLS(long[1:]/vol, DFNew).fit()
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
RegSpreads = OLS(spreads[1:], pd.DataFrame({'const' : 1, 'lag' : spreads[:-1]})).fit()
print(RegSpreads.summary())
plots(RegSpreads.resid, 'spreads')
RegVolSpreads = OLS(spreads[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : spreads[:-1]/vol, 'vol' : 1})).fit()
print(RegVolSpreads.summary())
plots(RegVolSpreads.resid, 'vol-spreads')

lspreads = np.log(spreads)
RegL = OLS(lspreads[1:], pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()
print(RegL.summary())
plots(RegL.resid, 'spreads')
RegLVol = OLS(lspreads[1:]/vol, pd.DataFrame({'const' : 1/vol, 'lag' : lspreads[:-1]/vol, 'vol' : 1})).fit()
print(RegLVol.summary())
plots(RegLVol.resid, 'vol-spreads')