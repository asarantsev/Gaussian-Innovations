The code and data for my-finance.org blogs https://my-finance.org/2026/05/09/improved-selection-of-model/ and https://my-finance.org/2026/05/09/improved-six-equation-model-selection/ 
where we select the time series model with rates and volatility, with and without the new valuation measure, for domestic and international stock returns, and for domestic corporate bonds. 

check-basic-model.py is checking these five-series models in https://my-finance.org/2026/05/09/improved-selection-of-model/

check-model.py is checking six-series model with the new valuation measure in https://my-finance.org/2026/05/09/improved-six-equation-model-selection/

simplest.py is checking the simplest model where normalized stock returns are IID Gaussian and normalized differences of log bond rates are also IID Gaussian

model0.py is simulating this simplest model

model1.py is simulating the more complicated 5-series model

model2.py is simulating the 6-series model

data2025.xlsx is the data file 1927--2025

5model.py and 6model.py are printing all these chosen regressions in 5-series and 6-series model

my_finance.py is a Python package when we take these simulated three series of annual returns for domestic stocks, international stocks, and domestic corporate bonds, as input, and create portfolio simulation based on that, and PDF/PNG graphs

test0.py is when we import model0.py and my_finance.py, and apply it to one example

test.py is importing model0.py, model1.py, model2.py, and my_finance.py, and apply it to three simple examples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spreads.xlsx is the raw data for long-short spreads, including Federal Reserve of Economic Data sources 

data2025-new.xlsx is the same data as in data2025.xlsx but including spread (only finalized data)

data2025-new.py is checking whether spreads follows an autoregression of order 1, with or without normalizing innovations by volatility, with all possible combinations of coefficients. Unfortunately, we conclude that our models with spreads do not pass these tests which we imposed in this blog post: https://my-finance.org/2026/05/09/improved-selection-of-model/

This is why we do not write any model with spreads. There is nothing to write here. I feel very disappointed at omitting such an important financial indicator. Maybe future research will show some improvement. 
