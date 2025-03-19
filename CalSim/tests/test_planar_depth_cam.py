#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np
import matplotlib.pyplot as plt

#system initial condition
x0 = np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T #start the quadrotor at 1 M in the air

#create an obstacle
qObs = np.array([[0, 1, 1], [0, 0.5, 2]]).T
rObs = [0.25, 0.25]
obstacleManager = cs.ObstacleManager(qObs, rObs, NumObs = 2)

#create a dynamics object for the double integrator
dynamics = cs.PlanarQrotor(x0)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create a depth camera
depthManager = cs.DepthCamManager(observerManager, obstacleManager, mean = None, sd = None)

#create a controller manager with a basic FF controller
controllerManager = cs.ControllerManager(observerManager, cs.FFController, depthManager = depthManager)

env = cs.Environment(dynamics, controllerManager, observerManager, obstacleManager)
env.reset()

env.run()

#get the pointcloud
ptcloudDict = depthManager.get_depth_cam_i(0).get_pointcloud()
ptcloudSpatial = ptcloudDict["ptcloud"]
y = ptcloudSpatial[1, :].tolist()
z = ptcloudSpatial[2, :].tolist()

#plot the pointcloud
plt.plot(y, z)
plt.show()