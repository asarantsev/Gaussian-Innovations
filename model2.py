import numpy

NSIMS = 400

Sigma = 0.0001 * numpy.array([[2.026369, 0.847703, 0.208415, -3.626994, -0.153965, 2.063568], [0.847703, 2.975298, -0.000237, -5.399599, 0.036818, 0.863375], [0.208415, -0.000237, 0.113497, 2.754026, -0.050806, 0.261286], [-3.626994, -5.399599, 2.754026, 1338.685234, 14.068448, -5.842714], [-0.153965, 0.036818, -0.050806, 14.068448, 1.935904, -0.844710], [2.063568, 0.863375, 0.261286, -5.842714, -0.844710, 3.152435]])

# The main simulation function
# initialV, initialR, initialM are initial volatiltiy and rates
# T is time horizon in years
# returns three 2d arrays, each has rows which are time series simulations
# domestic stocks, international stocks, and corporate bonds

def sim(initialV, initialR, initialM, T):
    
     # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(6), Sigma, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns
    noiseIntl = noise[:, :, 1] # international stock returns
    noiseBonds = noise[:, :, 2] # corporate bond returns
    noiseVol = noise[:, :, 3] # volatility 
    noiseRates = noise[:, :, 4] # corporate bond rates
    noiseMeasure = noise[:, :, 5] # the new valuation measure
    
    # now initialize the 2d arrays corresponding to simulated series
    simRetUSA = numpy.zeros((T, NSIMS))
    simRetIntl = numpy.zeros((T, NSIMS))
    simRetBonds = numpy.zeros((T, NSIMS))
    simLVol = numpy.zeros((T+1, NSIMS))
    simLRates = numpy.zeros((T+1, NSIMS))
    simMeasure = numpy.zeros((T+1, NSIMS))
    
    # initialize some simulated series given initial conditions
    simLVol[0] = numpy.log(initialV) * numpy.ones(NSIMS)
    simLRates[0] = numpy.log(initialR) * numpy.ones(NSIMS)
    simMeasure[0] = numpy.ones(NSIMS)
    
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
    
    # simulate the valuation measure as autoregression with stochastic volatility
    for t in range(T):
        simMeasure[t + 1] = 0.1699 + 0.8262 * simMeasure[t] - 0.0129 * simVol[t + 1] + simVol[t + 1] * noiseMeasure[t]
    
    # simulate three series of returns 
    for t in range(T):
        simRetUSA[t] = numpy.exp(0.2637 * numpy.ones(NSIMS) - 0.0129 * simVol[t+1] - 0.0553 * (simRates[t+1] - simRates[t]) - 0.1303 * simMeasure[t] + simVol[t + 1] * noiseUSA[t]) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(0.2684 * numpy.ones(NSIMS) - 0.018 * simVol[t+1] - 0.039 * (simRates[t+1] - simRates[t]) + simVol[t+1] * noiseIntl[t]) - numpy.ones(NSIMS)
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(- 0.0596 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
    
    return [simRetUSA, simRetIntl, simRetBonds]
