# Bayesian with transformed log volatility and transformed rate

import numpy as np
from scipy.stats import wishart
from scipy.stats import yeojohnson as YJ

# covariance matrix of five noise terms
C = 10**(-4) * np.array([[39.20, 7.74, -2.87, 1.63, -3.45], [7.74, 18.66, -1.07, -2.29, -0.35], [-2.87, -1.07, 2.36, 1.45, 1.09], [1.63, -2.29, 1.45, 7.18, -0.08], [-3.45, -0.35, 1.09, -0.08, 3.29]])
NSIMS = 10**4

covVol = np.linalg.inv(np.array([[97, 85.34], [85.34, 75.66]]))
covRate = np.linalg.inv(np.array([[98, 122.5], [122.5, 154.8]]))
covStock = np.linalg.inv(np.array([[1.592, 0.097], [0.097, 1.082]]))
covBond = np.linalg.inv(np.array([[53, 2.03], [2.03, 68.51]]))
covIntl = np.linalg.inv(np.array([[0.8416, 0.1902], [0.1902, 0.8903]]))
N = 98

# inverse Yeo-Johnson transform with index -0.5
def YJinv(y):
    return np.select([y < 0, (y >= 0) & (y < 1.6), y >= 1.6], [y, (1 - 0.5 * y)**(-2) - 1, 24 + 125 * (y - 1.6)])
    
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
    
    simTVol[0] = initialTVol * np.ones(NSIMS)
    simTRate[0] = initialTRate * np.ones(NSIMS)

    Wsamples = np.linalg.inv(wishart.rvs(df = N, scale = np.linalg.inv(N*C), size = NSIMS))
        
    coeffNoiseVol = np.random.multivariate_normal([0, 0], covVol, NSIMS)
    coeffNoiseRate = np.random.multivariate_normal([0, 0], covRate, NSIMS)
    coeffNoiseBond = np.random.multivariate_normal([0, 0], covBond, NSIMS)
    coeffNoiseIntl = np.random.multivariate_normal([0, 0], covIntl, NSIMS)
    coeffNoiseStock = np.random.multivariate_normal([0, 0], covStock, NSIMS)
    simConstVol = 0.3515 * np.ones(NSIMS) + coeffNoiseVol[:, 0] * np.sqrt(Wsamples[:, 0, 0])
    simLagVol = -0.3991 * np.ones(NSIMS) + coeffNoiseVol[:, 1] * np.sqrt(Wsamples[:, 0, 0])
    simConstRate = 0.0742 * np.ones(NSIMS) + coeffNoiseRate[:, 0] * np.sqrt(Wsamples[:, 1, 1])
    simLagRate = -0.0591 * np.ones(NSIMS) #+ coeffNoiseRate[:, 1] * np.sqrt(Wsamples[:, 1, 1])
    simConstStock = 0.1334 * np.ones(NSIMS) + coeffNoiseStock[:, 0] * np.sqrt(Wsamples[:, 2, 2])
    simDurationStock = 0.0658 * np.ones(NSIMS) + coeffNoiseStock[:, 1] * np.sqrt(Wsamples[:, 2, 2])
    simConstBond = -0.0159 * np.ones(NSIMS) + coeffNoiseBond[:, 0] * np.sqrt(Wsamples[:, 3, 3])
    simDurationBond = -0.0558 * np.ones(NSIMS) + coeffNoiseBond[:, 1] * np.sqrt(Wsamples[:, 3, 3])
    simConstIntl = 0.1271 * np.ones(NSIMS) + coeffNoiseIntl[:, 0] * np.sqrt(Wsamples[:, 4, 4])
    simDurationIntl = 0.0511 * np.ones(NSIMS) + coeffNoiseIntl[:, 1] * np.sqrt(Wsamples[:, 4, 4])
    
    noise = np.zeros((5, T, NSIMS))
    for sim in range(NSIMS):
        noise[:, :, sim] = np.transpose(np.random.multivariate_normal(np.zeros(5), Wsamples[sim], T))
    noiseVol = noise[0]
    noiseRate = noise[1]
    noiseStock = noise[2]
    noiseBond = noise[3]
    noiseIntl = noise[4]
        
    for t in range(T):
        simTVol[t + 1] = simTVol[t] * (np.ones(NSIMS) + simLagVol) + simConstVol + noiseVol[t]
    simVol = np.exp(YJinv(simTVol))
    for t in range(T):
        simTRate[t + 1] = simTRate[t] * (np.ones(NSIMS) + simLagRate) + simConstRate + noiseRate[t]
    simRate = YJinv(simTRate)
                
    for t in range(T):
        simGBond[t] = 0.01 * simRate[t] + simDurationBond * (simRate[t+1] - simRate[t]) + simConstBond + noiseBond[t]
        simGStock[t] = simConstStock - simDurationStock * (simRate[t+1] - simRate[t]) + simVol[t+1] * noiseStock[t]
        simGIntl[t] = simConstIntl - simDurationIntl * (simRate[t+1] - simRate[t]) + simVol[t+1] * noiseIntl[t]
    
    simBond = np.exp(simGBond) - np.ones((T, NSIMS))
    simStock = np.exp(simGStock) - np.ones((T, NSIMS))
    simIntl = np.exp(simGIntl) - np.ones((T, NSIMS))
        
    return [simStock, simIntl, simBond]