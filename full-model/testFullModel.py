from simFullModel import sim
from appFullModel import output

T = 30
initialW = 1
initialFlow = 0
growthFlow = 0.03
initialV = 10
initialR = 4
initialH = 0
initialL = 2.5

simulation = sim(initialV, initialH, initialR, initialL, T)
output(simulation, initialW, initialFlow, growthFlow, T, 0.1, 0.3, 0.6, 0.5, 0.5, initialV, initialH, initialR, initialL, 'output-graph')
