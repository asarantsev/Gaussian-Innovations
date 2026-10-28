from bayes import simBayes
from classic import sim
from app import output

T = 30
initialW = 1
initialFlow = 0
growthFlow = 0.03
initialV = 10
initialR = 4
bondStart = 0.1
bondEnd = 0.4
intlShare = 0.3

simulation = sim(initialV, initialR, T)
output(simulation, initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, initialV, initialR, 'output-classic')
bayesian = simBayes(initialV, initialR, T)
output(bayesian, initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, initialV, initialR, 'output-bayes')
