from bayes import simBayes
from classic import sim
from app import output

T = 40
initialW = 1000
initialFlow = -40
growthFlow = 0.03
initialV = 10
initialR = 4
bondStart = 0.1
bondEnd = 0.4
intlShare = 0.3

output(sim(initialV, initialR, T), initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, initialV, initialR, 'output-classic')
output(simBayes(initialV, initialR, T), initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, initialV, initialR, 'output-bayes')
