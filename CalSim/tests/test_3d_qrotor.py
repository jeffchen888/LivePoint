#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np

#system initial condition
pos0 = np.array([[0, 0, 1]]).T
vel0 = np.zeros((3, 1))
omega0 = np.zeros((3, 1))
R0 = np.eye(3).reshape((9, 1))
x0 = np.vstack((pos0, R0, omega0, vel0))

#create a dynamics object for the double integrator
dynamics = cs.Qrotor3D(x0)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create a controller manager with a basic FF controller
controllerManager = cs.ControllerManager(observerManager, cs.FFController)

env = cs.Environment(dynamics, controllerManager, observerManager)
env.reset()

#run the simulation
env.run()