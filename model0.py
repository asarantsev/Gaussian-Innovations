import numpy

# number of Monte Carlo simulations
NSIMS = 400

# covariance matrix of innovations 
Sigma = (10**(-4)) * numpy.array([[3.897114, 2.470823, -41.524404, -1.216955, 0.175224], [2.470823, 4.786725, -43.786001, -1.314104, -0.058867], [-41.524404, -43.786001, 1338.685234, 14.895365, 1.379118], [-1.216955, -1.314104, 14.895365, 1.999587, -0.025363], [0.175224, -0.058867, 1.379118, -0.025363, 0.102879]])

# The main simulation function of the simplest model
# initialV and initialR are initial volatiltiy and rates
# T is time horizon in years
# returns three 2d arrays, each has rows which are time series simulations
# domestic stocks, international stocks, and corporate bonds

# Equations are: for W_k(t) = wealth at end of year t, k = 0, 1, 2 for USA corporate bonds, USA stocks, international stocks
# (ln W_1(t) - ln W_1(t-1))/V(t) = m_1 + Z_1(t) 
# (ln W_2(t) - ln W_2(t-1))/V(t) = m_2 + Z_2(t) 
# ln(W_0(t)/W_0(t-1) - 0.01R(t-1)) = -d_0(R(t) - R(t-1)) + V(t)Z_0(t)
# ln V(t) = a + b * ln V(t-1) + Z_V(t)
# ln R(t) - ln R(t-1) = V(t)Z_R(t)

# The average long-term volatility V is 10.51
# The current (2025) volatility V is 11.77
# The average long-term BAA rate R is 6.8
# The current (December 2025) BAA rate is 5.9

def sim(initialV, initialR, T):
    
    # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(5), Sigma, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns Z_1
    noiseIntl = noise[:, :, 1] # international stock returns Z_2
    noiseVol = noise[:, :, 2] # volatility Z_V
    noiseRates = noise[:, :, 3] # corporate bond rates Z_R
    noiseBonds = noise[:, :, 4] # corporate bond returns Z_0
    
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
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(- 0.016269 * numpy.ones(NSIMS) - 0.062568 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
        
    return [simRetUSA, simRetIntl, simRetBonds]
