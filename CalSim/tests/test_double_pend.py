#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np
import matplotlib.pyplot as plt

#define an initial condition
q0 = np.array([[0.5, 0.5, 0.1, 0.1]]).T

#create a double pendulum
dynamics = cs.DoublePendulum(q0, N = 1)

#create a simulation environment
T = 30 #10 second simulation
env = cs.Environment(dynamics, None, None, T = T)

#run the simulation
xHist, uHist, tHist = env.run()

#extract q1, q2
q1Hist = xHist[0, :].tolist()
q2Hist = xHist[1, :].tolist()
q1DotHist = xHist[2, :].tolist()
q2DotHist = xHist[3, :].tolist()
tHist = tHist[0, :].tolist()

#plot q1 against q2
plt.plot(q1Hist, q2Hist)
plt.plot(q1DotHist, q2DotHist)
plt.show()
