from sim-full-model import sim
from app-full-model import output

T = 30
initialW = 1
initialFlow = -0.04
growthFlow = 0.03
initialV = 10
initialR = 4
initialM = 0
initialL = 2.5

simulation = sim(initialV, initialR, initialM, initialL, T)
output(simulation, initialW, initialFlow, growthFlow, T, 0.1, 0.3, 0.6, 0.5, 0.5, 'output-graph')
