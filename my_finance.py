# This package allows us to simulate portfolios and plot Monte Carlo simulations
# given that we already have simulated three asset classes
# USA stocks, international stocks, and USA corporate bonds

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

NSIMS = 400

# number of displayed simulations on the graph
NDISPLAYS = 5

# and selected percentages
selectedPercentages = [0.1, 0.3, 0.5, 0.7, 0.9]

# simulate wealth process given initial wealth and contributions/withdrawals
# simReturns is the 3d array of stock returns, domestic and international, and bond returns
# we assume we already simulated these asset classes from the previous model
# bondStart and bondEnd = share of bonds at the start and end of portfolio
# we allow stock/bond allocation to change in time
# intlShare = share of international in the stock part of the portfolio
# nYears = number of years, please make it an integer between 1 and 40
# initialW = initialWealth invested in this portfolio, please make it between 1 and 1000
# initialFlow (signed value) = first year flow: contribution (+) or withdrawal (-)
# growthFlow (signed value) = annual growth (+) or decline (-) of flow

def simWealth(simReturns, initialW, initialFlow, growthFlow, nYears, bondStart, bondEnd, intlShare):
    
    # Get returns of individual asset classes from overall returns
    simRetUSA = simReturns[0]
    simRetIntl = simReturns[1]
    simRetBonds = simReturns[2]
        
    # simulate the stock part of the portfolio
    simRetStock = simRetIntl * intlShare + simRetUSA * (1 - intlShare)
    
    # changing decomposition of stock/bond split
    bondP = np.linspace(bondStart, bondEnd, nYears)
    stockP = np.ones(nYears) - bondP
    simPortfolio = np.zeros((nYears, NSIMS))
    for t in range(nYears):
        simPortfolio[t] = bondP[t] * simRetBonds[t] + stockP[t] * simRetStock[t]
    
    # average arithmetic returns over each simulation
    pathData = np.mean(simPortfolio, axis = 0)
    
    # create an array for wealth simulation
    wealth = np.zeros((nYears + 1, NSIMS))
    
    # initial wealth year 0 initialize
    wealth[0] = np.ones(NSIMS) * initialW 

    # create (deterministic) array for flow (contributions and withdrawals)
    # for each year in nYears, exponentially growing or decreasing
    flow = initialFlow * np.exp(np.array(range(nYears)) * np.log(1 + growthFlow))

    # this is the main function connecting wealth to returns and flow
    if initialFlow < 0:
        for sim in range(NSIMS):
            for t in range(nYears):
                
                # main equation connecting returns, flow, wealth
                wealth[t + 1, sim] = wealth[t, sim] * (1 + simPortfolio[t, sim]) + flow[t]
                if wealth[t + 1, sim] <= 0:
                    pathData[sim] = t + 1 # record ruin year in place of average returns
                    wealth[t + 1:, sim] = 0 # all future wealth is zero
                    break # and stop this particular simulation

    else: # if no withdrawals then we do not need to check for bankruptcy
        for t in range(nYears):
            # main equation connecting returns, flow, wealth
            wealth[t+1] = wealth[t] * (1 + simPortfolio[t]) + flow[t] * np.ones(NSIMS)

    # timeAvgRet = average total portfolio return array over each path
    # wealth = paths of wealth
    return pathData, wealth

# Percentage format for probability 'x' rounded to 2 decimal points
# for text in output picture legend say 45.33%
def percent(x):
    return str(round(100*x, 2)) + '%'

# Wealth amount format with K, M, B and one decimal point
# to simplify output and make legend less cluttered
# K = 1,000, M = 1,000,000, B = 1,000,000,000
def form(x):
    if x < 10**3:
        return f"{x:.1f}"
    if 10**3 <= x < 10**6: # 1.2K not 1236
        return f"{10**(-3)*x:.1f}K"
    if 10**6 <= x < 10**9: # 15.2M not 15192124
        return f"{10**(-6)*x:.1f}M"
    if 10**9 <= x: # 24.7B not 24694M
        return f"{10**(-9)*x:.1f}B"

# This function is necessary to make the same K, M, B for y-axis formatting
# for the plot of wealth evolution
def tickFormat(x, pos):
    return form(x)

# Vertical lines on the graph of simulations
def allTicks(horizon):
    if horizon < 10:
        return range(horizon + 1) # if less than 10 years make all lines visible
    else: # make a line visible every 5 years, including the start
        step = int(horizon/5) # how many lines with 5-year intervals
        if horizon - 5 * step > 2: # horizon = 14, then lines = 0, 5, 10, 14
            return np.append(np.array(range(6))*step, [horizon])
        else: # horizon = 12, then lines = 0, 5, 12
            return np.append(np.array(range(5))*step, [horizon])

# text for legend: setup part, where we explain in words the inputs
# output will be created after simulation in the next function
# need to print this in the legend to the right of the main picture
# to remind the investor about their inputs
def setupText(initialWealth, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare):

    # This part is text description of flow (contributions or withdrawals)
    # Initial value for year 1 and rate of annual increase/decrease
    if initialFlow == 0:
        initialFlowText = 'No regular contributions or withdrawals'
        growthText = ''
    # case when contributions
    if initialFlow > 0:
        initFlow = form(initialFlow)
        if growthFlow == 0: # no change in contributions from year to year
            initialFlowText = 'Constant contributions ' + initFlow
            growthText = ''
        else:
            initialFlowText = 'Initial contributions ' + initFlow
            if growthFlow > 0:
                growthText = 'annual increase in contributions ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'annual decrease in contributions ' + percent(abs(growthFlow))

    # case when withdrawals
    if initialFlow < 0:
        initFlow = form(abs(initialFlow))
        if growthFlow == 0: # no change in withdrawals from year to year
            initialFlowText = 'Constant withdrawals ' + initFlow
            growthText = ''
        else:
            initialFlowText = 'Initial withdrawals ' + initFlow
            if growthFlow > 0:
                growthText = 'annual increase in withdrawals ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'annual decrease in withdrawals ' + percent(abs(growthFlow))

    # text output explaining portfolio weights
    # for example 33% American: S&P 500 and 67% International: MSCI EAFE
    # At the start, 60% stocks and 40% bonds, at the end, 90% stocks and 10% bonds
    usText = 'Stocks: ' + percent(1 - intlShare) + ' American: S&P 500'
    intlText = 'and ' + percent(intlShare) + ' International: MSCI EAFE'
    initText = 'Portfolio: Stocks and USA corporate bonds'
    startText = 'At the start: ' + percent(1 - bondShare0) + ' Stocks ' + percent(bondShare0) + ' Bonds'
    endText = 'At the end: ' + percent(1 - bondShare1) + ' Stocks ' + percent(bondShare1) + ' Bonds'

    # number of simulations, convert to string
    simText = str(NSIMS) + ' Monte Carlo simulations'

    # number of years in time horizon
    timeText = 'Time Horizon: ' + str(timeHorizon) + ' years'

    # initial wealth
    initWealthText = 'Initial Wealth ' + form(initialWealth)

    # return all these texts combined
    texts = [simText, usText, intlText, initText, startText, endText, timeText, initWealthText, initialFlowText, growthText]

    # combine all these texts and return combined text
    return 'SETUP: ' + '\n'.join(texts)

# Function creating the graph when click Compute
# Perform simulations and draw them on a picture
# Select NDISPLAYS paths ranked by final wealth
# This includes paths which end in zero wealth (ruin, bankruptcy)
# Write a legend for each path, and the overall legend
# Including setup in the above function and the results

def output(simReturns, initialW, initialFlow, growthFlow, timeHorizon, bondStart, bondEnd, intlShare):

    pathData, paths = simWealth(simReturns, initialW, initialFlow, growthFlow, timeHorizon, bondStart, bondEnd, intlShare)

    # take average total portfolio return over each path
    # and pick paths which do not end in bankruptcy (ruin)
    # average these averaged return over such paths
    allAvgReturns = [pathData[sim] for sim in range(NSIMS) if paths[-1, sim] > 0]
    
    # pick paths which do end in ruin and average ruin time over them
    allRuinTimes = [pathData[sim] for sim in range(NSIMS) if paths[-1, sim] == 0]

    if len(allAvgReturns) > 0:
        avgRet = np.mean(allAvgReturns)
        AvgRetText = 'average over paths without ruin time averaged returns: ' + percent(avgRet)
    if len(allAvgReturns) == 0:
        AvgRetText = 'all paths end in bankruptcy'
    if len(allRuinTimes) > 0:
        avgTime = np.mean(allRuinTimes)
        AvgRuinText = 'average ruin time for paths in ruin: ' + str(round(avgTime, 2))
    if len(allRuinTimes) == 0:
        AvgRuinText = 'no paths end in bankruptcy'

    wealthMean = np.mean(paths[-1]) # average final wealth over paths

    # share of paths which end in bankruptcy (ruin) = zero wealth
    ruinProb = np.mean([paths[-1, sim] == 0 for sim in range(NSIMS)])

    # sort all paths by final wealth from bottom to top
    sortedIndices = np.argsort(paths[-1])

    # indices for selected paths ranked by final wealth
    # NDISPLAYS = number of displayed paths on the main image
    # equidistant by ranks of final wealth
    selectedIndices = [sortedIndices[int(NSIMS*item)] for item in selectedPercentages]

    # all time points: 0, 1, ..., timeHorizon
    times = range(timeHorizon + 1)
    RuinProbText = 'Ruin Probability ' + percent(ruinProb)
    MeanText = 'average final wealth ' + form(wealthMean)
    ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + AvgRuinText

    # text for setup which is in the main legend for the plot
    # so that user sees output image in a different page than inputs
    # and does not forget these inputs
    SetupText = setupText(initialW, initialFlow, growthFlow, timeHorizon, bondStart, bondEnd, intlShare)

    # this plot of only one point in white color is necessary for big legend
    # because it serves as its anchor
    plt.plot([0], [initialW], color = 'w', label = SetupText + '\n\n' + ResultText)

    # next show plots of wealth paths
    for display in range(NDISPLAYS):
        index = selectedIndices[display]

        # text shows final wealth and its % rank
        rankText = ' final wealth, ranked ' + percent(selectedPercentages[display])
        endWealth = paths[-1][index]

        if (endWealth == 0): # this path ended with zero wealth
            pathLabel = '0' + rankText + '; bankruptcy in ' + str(int(pathData[index])) + ' years'
        else: # this path ends with positive wealth
            pathLabel = form(endWealth) + rankText + '; time averaged returns: ' + percent(pathData[index])
        plt.plot(times, paths[:, index], label = pathLabel) # plot this given path

    plt.gca().set_facecolor('ivory') # background plot color
    plt.xlabel('Years') # label of the X-axis, for time

    ticks = allTicks(timeHorizon) # make vertical lines selected years
    plt.xticks(ticks) # this uses the x-axis for these vertical lines
    plt.gca().set_ylabel('Wealth')

    # and for the y-axis, we format it according to the K, M, B format
    # using the function 'form' and 'tickFormat' above
    plt.gca().yaxis.set_major_formatter(FuncFormatter(tickFormat))

    plt.title('Wealth Plot') # title of the entire figure
    # properties of legend: location relative to the anchor above
    # font size and background color
    plt.legend(bbox_to_anchor = (1, 1.1), loc = 'upper left', prop = {'size': 12}, facecolor = 'azure')
    plt.grid(True) # make vertical and horizontal grid

    # save to folder 'static' to present in output page below
    plt.savefig('output.png', bbox_inches = 'tight')
    plt.savefig('output.pdf', bbox_inches = 'tight', format = 'pdf')
    plt.close()