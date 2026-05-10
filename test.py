import my_finance
import model0
import model1
import model2

T = 30
initialW = 1
initialFlow = -0.04
growthFlow = 0.03
initialVol = 10
initialRate = 4
initialMeasure = 0

simulation0 = model0.sim(initialVol, initialRate, T)
my_finance.output(simulation0, initialW, initialFlow, growthFlow, T, 0.1, 0.3, 0.6, 'output0')
simulation1 = model1.sim(initialVol, initialRate, T)
my_finance.output(simulation1, initialW, initialFlow, growthFlow, T, 0.1, 0.3, 0.6, 'output1')
simulation2 = model2.sim(initialVol, initialRate, initialMeasure, T)
my_finance.output(simulation2, initialW, initialFlow, growthFlow, T, 0, 0, 0.6, 'output2')