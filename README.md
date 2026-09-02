The code and data for my-finance.org foundational blog posts 

https://my-finance.org/2026/08/14/including-bond-factors-into-the-new-gaussian-model/ 

which discusses the 5 asset classes with 4 market factors model. We do this in full-model folder of this repository. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Also, this GitHub repository discusses https://my-finance.org/2026/05/09/improved-selection-of-model/ and https://my-finance.org/2026/05/09/improved-six-equation-model-selection/ where we select the time series model with rates and volatility, with and without the new valuation measure https://my-finance.org/2026/01/30/new-valuation-measure-based-on-dividends/ for domestic and international stock returns, and for domestic corporate bonds.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

check-basic-model.py is checking these five-series models in https://my-finance.org/2026/05/09/improved-selection-of-model/ and check-model.py is checking six-series model with the new valuation measure in https://my-finance.org/2026/05/09/improved-six-equation-model-selection/ and simplest.py is checking the simplest model where normalized stock returns are IID Gaussian and normalized differences of log bond rates are also IID Gaussian

model0.py is simulating this simplest model. model1.py is simulating the more complicated 5-series model. model2.py is simulating the 6-series model. data.xlsx is the data file 1927--2025 including long-term Treasury rates (which are not used in these models).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

5model.py and 6model.py are printing all these chosen regressions in 5-series and 6-series model. my_finance.py is a Python package when we take these simulated three series of annual returns for domestic stocks, international stocks, and domestic corporate bonds, as input, and create portfolio simulation based on that, and PDF/PNG graphs. test0.py is when we import model0.py and my_finance.py, and apply it to one example. test.py is importing model0.py, model1.py, model2.py, and my_finance.py, and apply it to three simple examples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We separately study long-short spreads in the subfolder spreads. The file there spreads.xlsx is the raw data for long-short spreads, including Federal Reserve of Economic Data sources. data2025.xlsx in this folder is the same data as in data2025.xlsx but including spread (only finalized data). data2025.py in this folder is checking whether spreads follows an autoregression of order 1, with or without normalizing innovations by volatility, with all possible combinations of coefficients. Unfortunately, we conclude that our models with spreads do not pass these tests which we imposed in this blog post: https://my-finance.org/2026/05/09/improved-selection-of-model/

This is why we do not write any model with spreads. There is nothing to write here. I feel very disappointed at omitting such an important financial indicator. Maybe future research will show some improvement

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Update: Consider fitting the updated BAA, Long rates, with December average. Data file is data.xlsx and we have three code files: 2d.py fits the updated (BAA, Long) data. We failed to make innovations Gaussian IID. log-rates.py is the code fitting the updated log(BAA), log(Long) models. We were able to fit only log(log(BAA)-log(Long)) as a simple autoregression, without volatility. risk-spread.py is the code fitting total normalized returns of S&P 500 vs spreads, and we can fit only the model vs spreads/vol, with or without logarithm. See the blog post https://my-finance.org/2026/05/25/including-bond-factors-into-the-new-gaussian-model/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

Update: Consider a 5-asset model with US stocks, international developed stocks, emerging markets stocks, 10-year Treasury bonds, and investment-grade corporate bonds. This is an improvement over our previous 3-asset model with US stocks, international developed stocks, and corporate bonds. Our new model has 4 factors: S&P volatility, the new valuation measure, BAA rate, and long-term Treasury rate 
(measured in this model using risk spread). Previously, we had only the former 3 factors (except the latter one). I created a subfolder full-model 

I united all data in completed file full-data.xlsx which includes long-term Treasury rates and emerging market returns. Also, I edited this data file by greatly improving the readme sheet of this spreadsheet. This helps us to fit this full model. Also, checkFullModel.py is verifying that the regressions in the above description work, in the sense that residuals are IID Gaussian. This is simply existing work from check-model.py and log-rates.py (the successful working part) plus emerging markets returns divided by volatility versus change in rates, so duration (the same as for developed markets). But we decided to put it in one Python file. 

Also, here we add spread to factors of normalized US stock geometric returns: We check which version of spread has the best predictive value: spread of rates, spread of log rates, log spread of rates, log spread of log rates. We pick the model with spread of rates. Also, we add spread and valuation to developed and emerging markets. We compare with cut regression, where we have only the duration factor, and valuation with risk spread are removed. But we choose the full regression. 

The file model8.py is similar to model5.py and model6.py where we have printed all regression results, and covariance and correlation matrix for all 8 residuals (4 market factors + 4 asset classes, except the long-term Treasury returns). For long-term Treasuries, their returns are deterministic functions of Treasury rates.

Finally, appFullModel.py is the simulation of these portfolio returns with these 5 asset classes, given that we already simulated these asset classes. This is an upgraded version of my_finance.py. 

All this is explained in an updated blog post https://my-finance.org/2026/08/14/including-bond-factors-into-the-new-gaussian-model/ see also another updated post https://my-finance.org/2026/08/14/market-models-with-gaussian-innovations/

We have the simulation file simFullModel.py similar to model0.py, model1.py, and model2.py which would be useful for building a financial app, together with the app-full-model.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Update: In this subfolder full-model, we get the Box-Cox transform testing for original and normalized series of log returns of the three stock asset classes. This corresponds to the blog post https://my-finance.org/2026/08/31/the-box-cox-transform-and-its-use-for-financial-data/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Update: In another subfolder treasury-bonds, we model total annual returns of 10-year zero-coupon Treasury bonds given rates of 10-year and 9-year zero-coupon Treasury rates, end-of-year. But these rates are close to 10-year coupon Treasury bonds, December average. Thus we can compute these from these 10-year coupon bonds. There is a small error, although the correlation between two series of returns is more than 97%. However, we still need to include these random terms, and therefore we have 9 equations and 9 innovation series. Not, as previously noted, 9 equations (5 for asset class returns + 4 for financial market factors) but only 8 innovation series. 
