#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np

#system initial condition
x0 = np.array([[0]]).T

#create a dynamics object for the double integrator
dynamics = cs.SimpleDiscrete(x0)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create a controller manager with a basic FF controller
controllerManager = cs.ControllerManager(observerManager, cs.FFController)

env = cs.Environment(dynamics, controllerManager, observerManager)
env.reset()

#run the simulation
env.run()