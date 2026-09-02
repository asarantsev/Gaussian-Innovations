import numpy
import pandas
import scipy
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.api import stats

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)

DF = pandas.read_excel('annual-zeros-1961.xlsx', sheet_name = 'summary')
benchmark = DF['Benchmark'].values
target = DF['Target'].values

plt.plot(benchmark, target, 'o')
plt.xlabel('Benchmark')
plt.ylabel('Target')
plt.show()
Years = range(1972, 2026)
plt.plot(Years, benchmark, label = 'Benchmark')
plt.plot(Years, target, label = 'Target')
plt.legend()
plt.show()

# Fit Regression
Reg = scipy.stats.linregress(benchmark, target)
resid = target - Reg.slope * benchmark - Reg.intercept * numpy.ones(54)
print(Reg)
verification(resid) 
plot_acf(resid)
plt.show()
plot_acf(abs(resid))
plt.show()
qqplot(resid, line = 's')
plt.show()

# Apply the Box-Cox transform to exponentiated residuals
nresid = scipy.stats.boxcox(numpy.exp(resid))[0]
print('lambda = ', scipy.stats.boxcox(numpy.exp(resid))[1])
verification(nresid)
plot_acf(nresid)
plt.show()
plot_acf(abs(nresid))
plt.show()
qqplot(nresid, line = 's')
plt.show()