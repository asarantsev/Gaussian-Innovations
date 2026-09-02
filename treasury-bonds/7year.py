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

DF = pandas.read_excel('annual-zeros-1961.xlsx', sheet_name = 'zeros')
benchmark = DF['Benchmark'].values[:-1]
target = DF['Target'].values[:-1]

plt.plot(benchmark, target, 'o')
plt.xlabel('Benchmark')
plt.ylabel('Target')
plt.show()

Years = range(1962, 2026)
plt.plot(Years, benchmark, label = 'Benchmark')
plt.plot(Years, target, label = 'Target')
plt.legend()
plt.show()

# Fit Regression
Reg = scipy.stats.linregress(benchmark, target)
resid = target - Reg.slope * benchmark - Reg.intercept * numpy.ones(64)
print(Reg)
verification(resid) 
plot_acf(resid)
plt.show()
plot_acf(abs(resid))
plt.show()
qqplot(resid, line = 's')
plt.show()