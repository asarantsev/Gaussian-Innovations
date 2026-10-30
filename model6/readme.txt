This subfolder contains the Python code to fit the 6-equation model: V = annual realized daily S&P volatility, 1928-2025; R = Bank of America ICE BAA rate, December daily average, 1927-2025; L = long-term (10-years) Treasury bond rates, December daily average, 1927-2025; F = Yeo-Johnson transform with index -0.5

F(ln V) and F(R) modeled each as an autoregression of order 1, but F(ln V) is mean-reverting and F(R) fail to reject unit root; also, S = ln(F(R) - F(L)) is an autoregression of order 1, mean-reverting.

geometric US and international stock returns are modeled as Q(t) = a - d(R(t) - R(t-1)) + V(t)Z(t); corporate bond returns are modeled as Q(t) - 0.01R(t-1) = a - d(R(t) - R(t-1)) + Z(t); Treasury bond returns are modeled using summation of coupons and principal, without random terms

Data: US stocks: S&P, 1928-2025; International stocks: 12% MSCI Canada + 88% MSCI EAFE, 1970-2025; Bonds: US BofA corporate bonds, 1973-2025.

All returns are annual (not monthly); total (including dividends); and nominal (not adjusted-inflation)

All six series of residuals are Gaussian, so we model this as multivariate Gaussian, which enables us to do objective Bayes inference

fit.py fitting all these regressions and printing Bayes covariance matrices of the posteriors; classic.py frequentist simulation; bayes.py objective Bayes simulation; app.py printing PDF and PNG output; test.py is execute all the above

Each has 400 simulations. We changed simulations by fixing Bayesian lag coefficient F(R) and not allowing prior there, since this leads to explosion. With this, even 10000 simulations do not explode, in both classic and Bayesian version. 

The data file data2025.xlsx is here, similar to the data file with the same name, the general folder but with improved readme sheet
