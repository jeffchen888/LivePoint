import CalSim as cs
import numpy as np
import yaml

#Before running, change desired config (doorway or intersection) file name to config.yaml
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

xD = config['xD']
xD1 = config['xD1']

pos0 = np.array([config['pos0']]).T
vel0 = np.zeros((3, 1))
omega0 = np.zeros((3, 1))
R0 = np.eye(3).reshape((9, 1))
x0 = np.vstack((pos0, R0, omega0, vel0))

pos1 = np.array([config['pos1']]).T
vel1 = np.zeros((3, 1))
omega1 = np.zeros((3, 1))
R1 = np.eye(3).reshape((9, 1))
x1 = np.vstack((pos1, R1, omega1, vel1))

dynamics = cs.Qrotor3D(x0)
dynamics1 = cs.Qrotor3D(x1)

observerManager = cs.ObserverManager(dynamics)
observerManager1 = cs.ObserverManager(dynamics1)

orig_qObs = np.array(config['orig_qObs']).T
orig_rObs = config['orig_rObs']
qObs = np.vstack((config['orig_qObs'], config['pos1'])).T
qObs1 = np.vstack((config['orig_qObs'], config['pos0'])).T
rObs = config['orig_rObs'] + [config['rObs']]
rObs1 = config['orig_rObs'] + [config['rObs1']]

orig_obstacleManager = cs.ObstacleManager(orig_qObs, orig_rObs, NumObs = config['numObs'])
obstacleManager = cs.ObstacleManager(qObs, rObs, NumObs = config['numObs'] + 1)
obstacleManager1 = cs.ObstacleManager(qObs1, rObs1, NumObs = config['numObs'] + 1)

depthManager = cs.DepthCamManager(observerManager, obstacleManager, mean = None, sd = None)
depthManager1 = cs.DepthCamManager(observerManager1, obstacleManager1, mean = None, sd = None)

xD = np.vstack((np.array([config['xD']]).T, R0, omega0, vel0))
xD1 = np.vstack((np.array([config['xD1']]).T, R1, omega1, vel1))

trajManager = cs.TrajectoryManager(x0, xD, Ts = 5, N = 1)
trajManager1 = cs.TrajectoryManager(x1, xD1, Ts = 5, N = 1)

controllerManager = cs.ControllerManager(observerManager, cs.QRotorGeometricPD, None, trajManager, depthManager)
controllerManager1 = cs.ControllerManager(observerManager1, cs.QRotorGeometricPD, None, trajManager1, depthManager1)

if config['liveness']:
    env = cs.EnvironmentWithLiveness(dynamics, dynamics1, controllerManager, controllerManager1,
                      observerManager, observerManager1, obstacleManager, obstacleManager1, orig_obstacleManager,
                      trajManager, trajManager1, depthManager, depthManager1, T = 10)
else: 
    env = cs.Environment(dynamics, dynamics1, controllerManager, controllerManager1,
                      observerManager, observerManager1, obstacleManager, obstacleManager1, orig_obstacleManager,
                      trajManager, trajManager1, depthManager, depthManager1, T = 10)
    
env.run()