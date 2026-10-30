# Classic with transformed log volatility
# and transformed rate

import numpy as np
from scipy.stats import yeojohnson as YJ

# covariance matrix of five noise terms
C = 10**(-4) * np.array([[39.20, 7.74, -2.87, 1.63, -3.45], [7.74, 18.66, -1.07, -2.29, -0.35], [-2.87, -1.07, 2.36, 1.45, 1.09], [1.63, -2.29, 1.45, 7.18, -0.08], [-3.45, -0.35, 1.09, -0.08, 3.29]])
NSIMS = 400

# inverse Yeo-Johnson transform with index -0.5
def YJinv(y):
    return np.select([y < 0, (y >= 0) & (y < 1.6), y >= 1.6], [y, (1 - 0.5 * y)**(-2) - 1, 24 + 125 * (y - 1.6)])

# main function to simulate returns of three asset classes
def sim(initialVol, initialRate, T):
    initialTRate = YJ(initialRate, -0.5) 
    initialTVol = YJ(np.log(initialVol), -0.5)
    
    # simulate 5-dim noise terms
    noise = np.random.multivariate_normal(np.zeros(5), C, (T, NSIMS))
    noiseVol = noise[:, :, 0]
    noiseRate = noise[:, :, 1]
    noiseStock = noise[:, :, 2]
    noiseBond = noise[:, :, 3]
    noiseIntl = noise[:, :, 4]
    
    # here we shall write the simulated factors and returns
    simStock = np.zeros((T, NSIMS))
    simBond = np.zeros((T, NSIMS))
    simIntl = np.zeros((T, NSIMS))
    simTVol = np.zeros((T+1, NSIMS))
    simTRate = np.zeros((T+1, NSIMS))
    
    # initialize the first volatility and rate after transform
    simTVol[0] = initialTVol * np.ones(NSIMS)
    simTRate[0] = initialTRate * np.ones(NSIMS)
    
    # run the actual simulation of transformed volatility
    for t in range(T):
        simTVol[t + 1] = simTVol[t] * (1 - 0.3991) + 0.3515 * np.ones(NSIMS) + noiseVol[t]
    simVol = np.exp(YJinv(simTVol)) # then transform it back
    
    # run the actual simulation of log rate
    for t in range(T):
        simTRate[t + 1] = 0.0742 * np.ones(NSIMS) + (1 - 0.0591) * simTRate[t] + noiseRate[t]   
    simRate = YJinv(simTRate) # then exponentiate to get rate
    
    # simulate the geometric returns of three asset classes
    for t in range(T): 
        simBond[t] = 0.01 * simRate[t] - 0.0558 * (simRate[t+1] - simRate[t]) - 0.0159 * np.ones(NSIMS) + noiseBond[t]
        simStock[t] = 0.1334 * np.ones(NSIMS) - 0.0658 * (simRate[t+1] - simRate[t]) + simVol[t + 1] * noiseStock[t]
        simIntl[t] = 0.1271 * np.ones(NSIMS) - 0.0511 * (simRate[t+1] - simRate[t]) + simVol[t + 1] * noiseIntl[t]
    
    # convert from geometric to arithmetic returns
    simABond = np.exp(simBond) - np.ones((T, NSIMS))
    simAStock = np.exp(simStock) - np.ones((T, NSIMS))
    simAIntl = np.exp(simIntl) - np.ones((T, NSIMS))
    
    return [simAStock, simAIntl, simABond]
