from bayes import simBayes
from classic import simClassic
from lite import simLite
from app import output

T = 40
initialW = 1000
initialFlow = -40
growthFlow = 0.03
initialV = 10
initialR = 8
initialL = 4.5
bondStart = 0.1
bondEnd = 0.4
intlShare = 0.3
trShare = 0.4

output(simClassic(initialV, initialR, initialL, T), initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, trShare, initialV, initialR, initialL, 'classic')
output(simBayes(initialV, initialR, initialL, T), initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, trShare, initialV, initialR, initialL, 'bayes')
output(simLite(initialV, initialR, initialL, T), initialW, initialFlow, growthFlow, T, bondStart, bondEnd, intlShare, trShare, initialV, initialR, initialL, 'lite')