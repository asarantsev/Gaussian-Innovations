import numpy

NSIMS = 400

Sigma = 0.0001 * numpy.array([[2.008, 0.872, 0.0630, 0.189, -4.717, -0.149, 2.047, -11.32], [0.872, 2.840, 2.022, 0.0124, -1.857, 0.08148, 0.9225, -15.84], [0.0630, 2.022, 5.731, -0.163, -5.364, 0.0795, 0.05017, -21.55], [0.189, 0.0124, -0.163, 0.1135, 2.754, -0.0508, 0.2613, 5.528], [-4.717, -1.857, -5.364, 2.754, 1339.0, 14.07, -5.843, 534.2], [-0.149, 0.08148, 0.0795, -0.0508, 14.07, 1.936, -0.8447, -4.628], [2.047, 0.9225, 0.05017, 0.2613, -5.843, -0.8447, 3.152, -7.534], [-11.32, -15.84, -21.55, 5.528, 534.2, -4.628, -7.534, 1166.0]])

# The main simulation function with the complete 9-series model
# 5 outputs: US stocks, ex-US developed stocks, emerging stocks
# and 10-year Treasury and investment-grade corporate bonds
# 4 factors: stock volatility V, and the valuation measure H
# BAA bond rates R, 10-year Treasury rates L 
# We use the valuation measure based on 1-year dividends
# initialF = initial factor (F = V, H, L, R)
# T is time horizon in years
# returns five 2d arrays, each has rows which are time series simulations

# geometric nominal returns 
# domestic stocks Q
# developed international stocks I
# and emerging stocks E

# arithmetic bond returns
# long-term Treasury A
# and investment-grade corporate bonds B

# Equations for geometric stock returns 

# Domestic: Q(t) = 0.2826 - 0.0110 * V(t) - 0.0609 * (R(t) - R(t-1)) -
# - 0.0173 * (R(t-1) - L(t-1)) - 0.1440 * H(t-1) + V(t) * Z_V(t)

# Developed international: I(t) = 0.2111 - 0.0196 * V(t) - 0.0316 * (R(t) - R(t-1)) -
# -0.0641 * H(t-1) + 0.0387 * (R(t-1) - L(t-1)) + V(t) * Z_I(t)

# Emerging: E(t) = 0.0544 - 0.0233 * V(t) - 0.0873* (R(t) - R(t-1)) -
# + 0.1155 * H(t-1) + 0.1051 * (R(t-1) - L(t-1)) + V(t) * V_E(t)

# Equations for arithmetic stock returns

# Investment-grade corporate bonds
# B(t) = 0.01 * R(t-1) + exp(-0.0596 * (R(t) - R(t-1)) + V(t) * Z_B(t)) - 1

# Treasury long-term (10-year) bonds
# A(t) = 1 + (1 + L(t)* 0.01)**(-9)/(1 + L(t-1)*0.01)**(-10)

# Next, the four factors are:
# with dC(t) = C(t) - C(t-1)

# Stock market volatility
# d(ln V(t)) = - 0.3824 * ln V(t-1) + 0.8569 + W_V(t)

# BAA rate
# d(ln R(t)) = - 0.0441 * ln R(t-1) + 0.0708 + V(t) * W_R(t)

# Spread of logs S(t) = ln R(t) - ln L(t)
# d(ln S(t)) = - 0.1873 * ln S(t-1) - 0.1980 + W_S(t)

# Valuation measure
# d H(t) = 0.1699 - 0.1738 * H(t-1) + V(t) * W_H(t)

# This is the main simulation function
# T = time horizon
def sim(initialV, initialH, initialR, initialL, T):
    
     # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(8), Sigma, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns Z_Q
    noiseIntl = noise[:, :, 1] # international developed stock returns Z_I
    noiseEm = noise[:, :, 2] # emerging stock returns Z_E
    noiseBonds = noise[:, :, 3] # corporate bond returns Z_B
    
    noiseVol = noise[:, :, 4] # volatility W_V
    noiseRates = noise[:, :, 5] # corporate bond rates W_R
    noiseMeasure = noise[:, :, 6] # the new valuation measure W_H
    noiseSpreads = noise[:, :, 7] # the log spreads of logs W_S
    
    # now initialize the 2d arrays corresponding to simulated series
    simRetUSA = numpy.zeros((T, NSIMS))
    simRetIntl = numpy.zeros((T, NSIMS))
    simRetEm = numpy.zeros((T, NSIMS))
    simRetBonds = numpy.zeros((T, NSIMS))
    simRetLong = numpy.zeros((T, NSIMS))
    
    simLVol = numpy.zeros((T+1, NSIMS))
    simLRates = numpy.zeros((T+1, NSIMS))
    simLSpreads = numpy.zeros((T+1, NSIMS))
    simMeasure = numpy.zeros((T+1, NSIMS))
        
    # initialize some simulated series given initial conditions
    simLVol[0] = numpy.log(initialV) * numpy.ones(NSIMS)
    simLRates[0] = numpy.log(initialR) * numpy.ones(NSIMS)
    simMeasure[0] = initialH * numpy.ones(NSIMS)
    simLSpreads[0] = numpy.log(numpy.log(initialR) - numpy.log(initialL)) * numpy.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = 0.8569 * numpy.ones(NSIMS) + (1 - 0.3824) * simLVol[t] + noiseVol[t]
        
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

    # simulate the log spread as autoregression
    for t in range(T):
        simLSpreads[t + 1] = (1 - 0.1873) * simLSpreads[t] - 0.1980 * numpy.ones(NSIMS) + noiseSpreads[t]
        
    # simulate the long-term Treasury rate
    # We rewrite the equation S(t) = ln (R(t)) - ln (L(t))
    # as L(t) = exp(ln R(t) - S(t))
    simLong = numpy.exp(simLRates - numpy.exp(simLSpreads))
    simSpread = simRates - simLong # and simulated spreads
    
    # simulate arithmetic stock returns 
    for t in range(T):
        # three series of stock returns
        simRetUSA[t] = numpy.exp(0.2826 * numpy.ones(NSIMS) - 0.0110 * simVol[t+1] - 0.0609 * (simRates[t+1] - simRates[t]) - 0.1440 * simMeasure[t] - 0.0173 * simSpread[t] + simVol[t + 1] * noiseUSA[t]) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(0.2111 * numpy.ones(NSIMS) - 0.0196 * simVol[t+1] - 0.0316 * (simRates[t+1] - simRates[t]) - 0.0641 * simMeasure[t] + 0.0387 * simSpread[t] + simVol[t+1] * noiseIntl[t]) - numpy.ones(NSIMS)
        simRetEm[t] = numpy.exp(0.0544 * numpy.ones(NSIMS) - 0.0233 * simVol[t+1] - 0.0873 * (simRates[t+1] - simRates[t]) + 0.1155 * simMeasure[t] + 0.1051 * simSpread[t] + simVol[t + 1] * noiseEm[t]) - numpy.ones(NSIMS)
        
        # two series of bond returns
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(- 0.0596 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
        simRetLong[t] = ((numpy.ones(NSIMS) + 0.01 * simLong[t])**10)*((numpy.ones(NSIMS) + 0.01 * simLong[t+1])**(-9)) - numpy.ones(NSIMS)
    
    return [simRetUSA, simRetIntl, simRetEm, simRetLong, simRetBonds]