import numpy

# number of Monte Carlo simulations
NSIMS = 400

# covariance matrix of innovations 
Sigma = (10**(-4)) * numpy.array([[2.218258, 0.948039, 0.215726, -6.113641, -0.210064], [0.948039, 2.975298, -0.000237, -5.399599, 0.036818], [0.215726, -0.000237, 0.113497, 2.754026, -0.050806], [-6.113641, -5.399599, 2.754026, 1338.685234, 14.068448], [-0.210064, 0.036818, -0.050806, 14.068448, 1.935904]])

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
    noiseBonds = noise[:, :, 2] # corporate bond returns
    noiseVol = noise[:, :, 3] # volatility 
    noiseRates = noise[:, :, 4] # corporate bond rates
    
    
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
        simLRates[t + 1] = 0.0708 + (1 - 0.0411) * simLRates[t] + noiseRates[t] * simVol[t + 1] 
        
    # take exponents to get rates
    simRates = numpy.exp(simLRates)
    
    # simulate three series of returns 
    for t in range(T):
        simRetUSA[t] = numpy.exp(0.2111 * numpy.ones(NSIMS) - 0.0107 * simVol[t+1] - 0.0621 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseUSA[t]) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(0.2684 * numpy.ones(NSIMS) - 0.018 * simVol[t+1] - 0.039 * (simRates[t+1] - simRates[t]) + simVol[t+1] * noiseIntl[t]) - numpy.ones(NSIMS)
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(- 0.0596 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
    
    return [simRetUSA, simRetIntl, simRetBonds]