# Bayesian, only for transformed rate
import numpy as np
from scipy.stats import yeojohnson as YJ

# covariance matrix of five noise terms
NSIMS = 10000
stdRate = np.sqrt(18.66) * 0.01
covRate = np.linalg.inv(np.array([[98, 122.5], [122.5, 154.8]]))
precision = stdRate ** (-2)
N = 98

# inverse Yeo-Johnson transform with index -0.5
def YJinv(y):
    x = np.empty_like(y)
    pos = y >= 0
    neg = ~pos
    x[pos] = np.power(1 + (-0.5) * y[pos], -2) - 1
    x[neg] = 1 - np.power(1 - (2.5) * y[neg], 0.4)
    return x 

def simBayes(initialRate, T):
    simPrec = (precision / (0.5*N)) * np.random.gamma(shape = 0.5*N, scale = 1, size = NSIMS)
    simStd = simPrec ** (-0.5)
    noise = np.zeros((T, NSIMS))
    for sim in range(NSIMS):
        noise[:, sim] = np.random.normal(0, stdRate, T)
    simRate = np.zeros((T+1, NSIMS))
    simTRate = np.zeros((T+1, NSIMS))
    initialTRate = YJ(initialRate, -0.5) 
    simTRate[0] = initialTRate * np.ones(NSIMS)
    coeffNoiseRate = np.random.multivariate_normal([0, 0], covRate, NSIMS)
    simConstRate = 0.0742 * np.ones(NSIMS) + coeffNoiseRate[:, 0] * simStd
    simLagRate = -0.0591 * np.ones(NSIMS) #+ coeffNoiseRate[:, 1] * simStd
    for t in range(T):
        simTRate[t + 1] = simTRate[t] * (np.ones(NSIMS) + simLagRate) + simConstRate + noise[t]
    simRate = YJinv(simTRate)
    return np.min(simTRate), np.max(simTRate)

print(simBayes(8, 40))