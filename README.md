The code and data for my-finance.org blogs 

https://my-finance.org/2026/05/09/improved-selection-of-model/

https://my-finance.org/2026/05/09/improved-six-equation-model-selection/

where we select the time series model with rates and volatility, with and without the new valuation measure, for domestic and international stock returns, and for domestic corporate bonds. 

check-basic-model.py is checking these five-series models in 

https://my-finance.org/2026/05/09/improved-selection-of-model/

simplest.py is checking the simplest model where normalized stock returns are IID Gaussian and normalized differences of log bond rates are also IID Gaussian

model0.py is simulating this simplest model

data2025.xlsx is the data file 1927--2025

check-model.py is checking six-series model with the new valuation measure in 

https://my-finance.org/2026/05/09/improved-six-equation-model-selection/

5model.py and 6model.py are printing all these chosen regressions in 5-series and 6-series model

my_finance.py is a Python package when we take these simulated three series of annual returns for domestic stocks, international stocks, and domestic corporate bonds, as input, and create portfolio simulation based on that, and PDF/PNG graphs

