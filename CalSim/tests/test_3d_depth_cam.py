#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#system initial condition
pos0 = np.array([[0, 0, 1]]).T
vel0 = np.zeros((3, 1))
omega0 = np.zeros((3, 1))
R0 = np.eye(3).reshape((9, 1))
x0 = np.vstack((pos0, R0, omega0, vel0))

#create a dynamics object for the 3D quadrotor
dynamics = cs.Qrotor3D(x0)

#create an obstacle
qObs = np.array([[0, 1, 1], [0, 0.5, 2]]).T
rObs = [0.25, 0.25]
obstacleManager = cs.ObstacleManager(qObs, rObs, NumObs = 2)

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
x = ptcloudSpatial[0, :].tolist()
y = ptcloudSpatial[1, :].tolist()
z = ptcloudSpatial[2, :].tolist()

#9800 points with two obstacles
print("Number of Points:", len(z))

#plot the pointcloud
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z)
plt.title("Quadrotor Trajectory")
plt.show()