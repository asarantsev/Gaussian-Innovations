import numpy
from matplotlib import pyplot

# number of Monte Carlo simulations
NSIMS = 400

# covariance matrix of innovations 
Sigma = (10**(-4)) * numpy.array([[3.897114, 2.470823, -41.524404, -1.216955, 0.175224], [2.470823, 4.786725, -43.786001, -1.314104, -0.058867], [-41.524404, -43.786001, 1338.685234, 14.895365, 1.379118], [-1.216955, -1.314104, 14.895365, 1.999587, -0.025363], [0.175224, -0.058867, 1.379118, -0.025363, 0.102879]])

# The main simulation function
# initialV and initialR are initial volatiltiy and rates
# T is time horizon in years
# returns three 2d arrays, each has rows which are time series simulations
# domestic stocks, international stocks, and corporate bonds

def sim(initialV, initialR, T):
    
    # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(5), Sigma, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns
    noiseIntl = noise[:, :, 1] # international stock returns
    noiseVol = noise[:, :, 2] # volatility 
    noiseRates = noise[:, :, 3] # corporate bond rates
    noiseBonds = noise[:, :, 4] # corporate bond returns
    
    # now initialize the 2d arrays corresponding to simulated series
    simLVol = numpy.zeros((T+1, NSIMS))
    simLRates = numpy.zeros((T+1, NSIMS))
    simRetUSA = numpy.zeros((T, NSIMS))
    simRetIntl = numpy.zeros((T, NSIMS))
    simRetBonds = numpy.zeros((T, NSIMS))
    
    # initialize some simulated series given initial conditions
    simLVol[0] = numpy.log(initialV) * numpy.ones(NSIMS)
    simLRates[0] = numpy.log(initialR) * numpy.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = 0.8569 * numpy.ones(NSIMS) + 0.6176 * simLVol[t] + noiseVol[t]
        
    # take exponents to get volatility
    simVol = numpy.exp(simLVol)
    
    # simulate log rates as heteroscedastic random walk
    for t in range(T):
        simLRates[t + 1] = noiseRates[t] * simVol[t + 1] + simLRates[t]
        
    # take exponents to get rates
    simRates = numpy.exp(simLRates)
    
    # simulate three series of returns 
    for t in range(T):
        simRetUSA[t] = numpy.exp(simVol[t + 1] * (noiseUSA[t] + 0.014325 * numpy.ones(NSIMS))) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(simVol[t + 1] * (noiseIntl[t] + 0.013599 * numpy.ones(NSIMS))) - numpy.ones(NSIMS)
        simRetBonds[t] = numpy.exp(0.01 * simRates[t] - 0.016269 * numpy.ones(NSIMS) - 0.062568 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
    return [simRetUSA, simRetIntl, simRetBonds]