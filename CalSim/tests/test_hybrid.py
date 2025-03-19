#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np

#system initial condition
x0 = np.array([[1.5, 0]]).T

#create a dynamics object for the bouncing ball/simple two-domain system
# dynamics = cs.BouncingBall(x0)
dynamics = cs.SimpleTwoDomainHybrid(x0)

#reset environment
env = cs.Environment(dynamics)
env.reset()

#run the simulation
env.run()