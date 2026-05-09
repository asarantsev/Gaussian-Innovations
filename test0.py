import my_finance
import model0

T = 40
initialVol = 10
initialRate = 9

simulation = model0.sim(initialVol, initialRate, T)
my_finance.output(simulation, 10, -0.5, 0.03, T, 0.3, 0.8, 0.6)