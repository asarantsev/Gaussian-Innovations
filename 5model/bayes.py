# Bayesian with transformed log volatility and transformed rate

import numpy as np
from scipy.stats import wishart
from scipy.stats import yeojohnson as YJ
np.random.seed(0)

# covariance matrix of five noise terms
C = 10**(-4) * np.array([[39.20, 7.74, -2.87, 1.63, -3.45], [7.74, 18.66, -1.07, -2.29, -0.35], [-2.87, -1.07, 2.36, 1.45, 1.09], [1.63, -2.29, 1.45, 7.18, -0.08], [-3.45, -0.35, 1.09, -0.08, 3.29]])
NSIMS = 10000
covVol = np.linalg.inv(np.array([[97, 85.34], [85.34, 75.66]]))
covRate = np.linalg.inv(np.array([[98, 122.5,], [122.5, 154.8]]))
covStock = np.linalg.inv(np.array([[1.592, 0.097], [0.097, 1.082]]))
covBond = np.linalg.inv(np.array([[53, 2.03], [2.03, 68.51]]))
covIntl = np.linalg.inv(np.array([[0.8416, 0.1902], [0.1902, 0.8903]]))
N = 98

# inverse Yeo-Johnson transform with index -0.5
def YJinv(y):
    x = np.empty_like(y)
    pos = y >= 0
    neg = ~pos
    x[pos] = np.power(1 + (-0.5) * y[pos], -2) - 1
    x[neg] = 1 - np.power(1 - (2.5) * y[neg], 0.4)
    return x 

# main function to simulate returns of three asset classes
def simBayes(initialVol, initialRate, T):
    
    # here we shall write the simulated factors and returns
    simGStock = np.zeros((T, NSIMS))
    simGBond = np.zeros((T, NSIMS))
    simGIntl = np.zeros((T, NSIMS))
    simVol = np.zeros((T+1, NSIMS))
    simRate = np.zeros((T+1, NSIMS))
    
    simTVol = np.zeros((T+1, NSIMS))
    simTRate = np.zeros((T+1, NSIMS))
    
    initialTRate = YJ(initialRate, -0.5) 
    initialTVol = YJ(np.log(initialVol), -0.5)
    
    Wsamples = np.linalg.inv(wishart.rvs(df = N, scale = np.linalg.inv(N*C), size = NSIMS))

    coeffNoiseVol = np.random.multivariate_normal([0, 0], covVol, NSIMS)
    coeffNoiseRate = np.random.multivariate_normal([0, 0], covRate, NSIMS)
    coeffNoiseBond = np.random.multivariate_normal([0, 0], covBond, NSIMS)
    coeffNoiseIntl = np.random.multivariate_normal([0, 0], covIntl, NSIMS)
    coeffNoiseStock = np.random.multivariate_normal([0, 0], covStock, NSIMS)

    for sim in range(NSIMS):
        simCoeffVol = np.array([0.3515, -0.3991]) + coeffNoiseVol[sim] * np.sqrt(Wsamples[sim, 0, 0])
        simConstVol = simCoeffVol[0]
        simLagVol = simCoeffVol[1]
        simCoeffRate = np.array([0.0742, -0.0591]) + coeffNoiseRate[sim] * np.sqrt(Wsamples[sim, 1, 1])
        simConstRate = simCoeffRate[0]
        simLagRate = simCoeffRate[1]
        simCoeffStock = np.array([0.1334, 0.0658]) + coeffNoiseStock[sim] * np.sqrt(Wsamples[sim, 2, 2])
        simConstStock = simCoeffStock[0]
        simDurationStock = simCoeffStock[1]
        simCoeffBond = np.array([-0.0159, 0.0558]) + coeffNoiseBond[sim] * np.sqrt(Wsamples[sim, 3, 3])
        simConstBond = simCoeffBond[0]
        simDurationBond = simCoeffBond[1]
        simCoeffIntl = np.array([0.1271, 0.0511]) + coeffNoiseIntl[sim] * np.sqrt(Wsamples[sim, 4, 4])
        simConstIntl = simCoeffIntl[0]
        simDurationIntl = simCoeffIntl[1]
        
        noise = np.random.multivariate_normal(np.zeros(5), Wsamples[sim], T)
        noiseVol = noise[:, 0]
        noiseRate = noise[:, 1]
        noiseStock = noise[:, 2]
        noiseBond = noise[:, 3]
        noiseIntl = noise[:, 4]
        
        for t in range(T):
            simTVol[t + 1, sim] = simTVol[t, sim] * (1 + simLagVol) + simConstVol + noiseVol[t]
        simVol[:, sim] = np.exp(YJinv(simTVol[:, sim]))
               
        for t in range(T):
            simTRate[t + 1, sim] = simTRate[t, sim] * (1 + simLagRate) + simConstRate + noiseRate[t]
        simRate[:, sim] = YJinv(simTRate[:, sim])
            
        simGBond[:, sim] = 0.01 * simRate[:-1, sim] - simDurationBond * np.diff(simRate[:, sim]) - simConstBond * np.ones(T) + noiseBond
        simGStock[:, sim] = simConstStock * np.ones(T) - simDurationStock * np.diff(simRate[:, sim]) + simVol[1:, sim] * noiseStock
        simGIntl[:, sim] = simConstIntl * np.ones(T) - simDurationIntl * np.diff(simRate[:, sim]) + simVol[1:, sim] * noiseIntl
        
    simBond = np.exp(simGBond) - np.ones((T, NSIMS))
    simStock = np.exp(simGStock) - np.ones((T, NSIMS))
    simIntl = np.exp(simGIntl) - np.ones((T, NSIMS))
        
    return [simStock, simIntl, simBond]

