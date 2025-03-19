"""
This file contains the core functionalities for the different Depth Cameras/LIDAR Sensors
"""
import numpy as np
from .dynamics import *

class PlanarDepthCam:
    def __init__(self, index, observerManager, obstacleManager, useXYZ = [True, True, False], mean = None, sd = None):
        """
        Init function for a quadrotor depth camera. Returns an XYZ pointcloud of the 
        positions of the obstacles in the environment in the frame of the ith quadrotor.
        Note that the pointcloud is generated ONLY in the YZ plane due to the planar nature
        of the system.

        Inputs:
            observerManager (ObserverManager): manager for state estimation of ith quadrotor
            obstacleManager (ObstacleManager): manager for the set of NumObs obstacles
            useXYZ (list): Select which indices to populate with nonzero values. This is a planar cam, so only two should be selected.
            mean (float): mean for measurement noise
            sd (float): standard deviation for measurement noise
        """

        #store input paramters
        self.index = index
        self.obstacleManager = obstacleManager
        self.useXYZ = useXYZ
        self.mean = mean
        self.sd = sd

        #extract the observer of quadrotor i
        self.observer = observerManager.get_observer_i(self.index)

        #store an attribute for the number of points to be sampled per obstacle
        self.numPts = 1000//self.obstacleManager.NumObs

        #store an attribute for the pointcloud data
        self._ptcloudData = {}

    def calc_ptcloud_world(self):
        """
        Calculate the pointcloud of the environment using the obstalce manager
        This is calculated in the world frame. As this is a planar camera
        the x coordinate is always zero - pointcloud is generated in YZ plane.
        Inputs:
            None
        Returns:
            3xN NumPy array: pointcloud of environment in the world frame
        """

        #initialize arrays for each coordinate
        xList = []
        yList = []
        zList = []
        for i in range(self.obstacleManager.NumObs):
            #extract the ith obstacle object
            obsI = self.obstacleManager.get_obstacle_i(i)

            #get the center (XYZ) and radius
            qCenter, r = obsI.get_center(), obsI.get_radius()

            #calculate a sample of points on the obstacle in the YZ plane
            for j in range(self.numPts):
                #calculate an angle theta to sample around the circle
                thetaJ = j*2*np.pi/self.numPts

                #calculate each coordinate
                xList.append(0)
                yList.append(qCenter[1, 0] + r*np.cos(thetaJ))
                zList.append(qCenter[2, 0] + r*np.sin(thetaJ))

        #check if this is an XY or YZ pointcloud:
        if self.useXYZ[0] and self.useXYZ[1]:
            #put the pointcloud in the XY plane
            return np.array([yList, zList, xList])
        else:
            #put the pointcloud in the YZ plane
            return np.array([xList, yList, zList])
    
    def compute_rotation(self):
        """
        Compute the rotation matrix from the agent frame to the world frame
        This should be implemented in instance of the depth camera.
        Returns:
            Rsq: rotation matrix between spatial and quadrotor frames
        """

        Rsq = np.eye(3)
        return Rsq
    
    def calc_ptcloud(self):
        """
        Calculates the pointcloud in the frame of agent i, updates the 
        self._ptcloudData attribute.
        """

        #use the observer to get the position and angle
        state = self.observer.get_state() #(x, y, z, theta, x_dot, y_dot, z_dot, theta_dot)

        #get position and angle of the system
        q = self.observer.get_pos()

        #compute rotation matrix to spatial frame
        Rsq = self.compute_rotation()

        #compute pointcloud in world frame
        ptcloudWorld = self.calc_ptcloud_world()

        #transform pointcloud
        ptcloudAgent = Rsq.T @ (ptcloudWorld - q)

        #store in the pointcloud attribute
        self._ptcloudData["ptcloud"] = ptcloudAgent
        self._ptcloudData['ptcloudworld'] = ptcloudWorld
        self._ptcloudData["stateVec"] = state
        return self._ptcloudData
    
    def get_pointcloud(self, update = True):
        """
        Returns the pointcloud dictionary from the class attribute 
        Args:
            update: whether or not to recalculate the pointcloud
        Returns:
            Dict: dictionary of pointcloud points and state vector at time of capture
        """

        #first, calculate the pointcloud
        if update:
            self.calc_ptcloud()
        return self._ptcloudData
    
class DepthCam3D(PlanarDepthCam):
    def __init__(self, index, observerManager, obstacleManager, mean = None, sd = None):
        """
        Init function for a quadrotor depth camera. Returns an XYZ pointcloud of the 
        positions of the obstacles in the environment in the frame of the ith quadrotor.
        Note that the pointcloud is generated ONLY in the YZ plane due to the planar nature
        of the system.

        Inputs:
            observerManager (ObserverManager): manager for state estimation of ith quadrotor
            obstacleManager (ObstacleManager): manager for the set of NumObs obstacles
            mean (float): mean for measurement noise
            sd (float): standard deviation for measurement noise
        """

        #call super init function -> set useXYZ to all true
        super().__init__(index, observerManager, obstacleManager, useXYZ = [True, True, True], mean = mean, sd = sd)

        #override number of points for a denser pointcloud
        self.numPts = 10000//self.obstacleManager.NumObs

    def calc_ptcloud_world(self):
        """
        Calculate the pointcloud of the environment using the obstacle manager
        This is calculated in the world frame. As this is a planar camera
        the x coordinate is always zero - pointcloud is generated in YZ plane.
        Inputs:
            None
        Returns:
            3xN NumPy array: pointcloud of environment in the world frame
        """
      
        #initialize arrays for each coordinate
        xList = []
        yList = []
        zList = []
        for i in range(self.obstacleManager.NumObs):
            #extract the ith obstacle object
            obsI = self.obstacleManager.get_obstacle_i(i)

            #get the center (XYZ) and radius
            qCenter, r = obsI.get_center(), obsI.get_radius()

            #generate a set of points in spherical coordinates
            gridDimn = int(self.numPts**0.5) #account for squaring of grid size
            thetaMesh = [j*2*np.pi/gridDimn for j in range(gridDimn)]
            phiMesh =  [j*2*np.pi/gridDimn for j in range(gridDimn)]

            #generate a meshgrid
            THETA, PHI = np.meshgrid(thetaMesh, phiMesh)

            #calculate a sample of points on the obstacle in the YZ plane
            for j in range(THETA.shape[0]):
                for k in range(THETA.shape[1]):
                    #Sample a tuple in spherical coordinates
                    theta, phi = THETA[j, k], PHI[j, k]

                    #calculate each coordinate
                    xList.append(qCenter[0, 0] + r*np.sin(theta)*np.cos(phi))
                    yList.append(qCenter[1, 0] + r*np.sin(theta)*np.sin(phi))
                    zList.append(qCenter[2, 0] + r*np.cos(theta))

        #return the resulting pointcloud
        return np.array([xList, yList, zList])
    
    def compute_rotation(self):
        """
        Compute the rotation matrix from the agent frame to the world frame
        This should be implemented in instance of the depth camera.
        Inputs:
            theta: angle of rotation about the x-axis -> note: this is not needed here.
        Returns:
            Rsq: rotation matrix between spatial and quadrotor frames
        """

        #retrieve and return the rotation matrix from the observer
        return self.observer.get_orient()
    
class PlanarDepthCamQrotor(PlanarDepthCam):
    def __init__(self, index, observerManager, obstacleManager, mean = None, sd = None):
        """
        Init function for a quadrotor depth camera. Returns an XYZ pointcloud of the 
        positions of the obstacles in the environment in the frame of the ith quadrotor.
        Note that the pointcloud is generated ONLY in the YZ plane due to the planar nature
        of the system.

        Inputs:
            observerManager (ObserverManager): manager for state estimation of ith quadrotor
            obstacleManager (ObstacleManager): manager for the set of NumObs obstacles
            mean (float): mean for measurement noise
            sd (float): standard deviation for measurement noise
        """
        #initialize the super function. Set useXYZ to [False, True, True] to store ptcloud in YZ
        super().__init__(index, observerManager, obstacleManager, useXYZ=[False, True, True], mean = mean, sd = sd)
    
    def compute_rotation(self):
        """
        Compute the rotation matrix from the quadrotor frame to the world frame
        Inputs:
            theta: angle of rotation about the x-axis
        Returns:
            Rsq: rotation matrix between spatial and quadrotor frames
        """
        #get the angle theta of the system
        theta = self.observer.get_orient()
        Rsq = np.array([[1, 0, 0], 
                        [0, np.cos(theta), -np.sin(theta)], 
                        [0, np.sin(theta), np.cos(theta)]])
        return Rsq
    
class PlanarDepthCamTurtlebot(PlanarDepthCam):
    def __init__(self, index, observerManager, obstacleManager, mean = None, sd = None):
        """
        Init function for a turtlebot depth camera. Returns an XYZ pointcloud of the 
        positions of the obstacles in the environment in the frame of the ith quadrotor.
        Note that the pointcloud is generated ONLY in the XY plane due to the planar nature
        of the system.

        Inputs:
            observerManager (ObserverManager): manager for state estimation of ith quadrotor
            obstacleManager (ObstacleManager): manager for the set of NumObs obstacles
            mean (float): mean for measurement noise
            sd (float): standard deviation for measurement noise
        """
        #call super init with useXYZ = [True, True, False]
        super().__init__(index, observerManager, obstacleManager, useXYZ = [True, True, False], mean = mean, sd = sd)

    def compute_rotation(self):
        """
        Compute the rotation matrix from the turtlebot frame to the world frame
        Returns:
            Rse: rotation matrix between spatial and ego frames
        """
        #get the angle phi of the system
        phi = self.observer.get_orient()
        Rse = np.array([[np.cos(phi), -np.sin(phi), 0], 
                        [np.sin(phi), np.cos(phi), 0], 
                        [0, 0, 1]])
        return Rse

class EgoLidar:
    def __init__(self, index, observerManager, mean = None, sd = None):
        """
        Lidar for an ego turtlebot. Returns a pointcloud containing
        other turtlebots in the environment.

        Args:
            index (int) : index of the ego turtlebot (zero-indexed from the first turtlebots)
            observerManager (ObserverManager) : Manager object for state estimation
            mean (float): mean for measurement noise
            sd (float): standard deviation for measurement noise
        """
        #store input parameters
        self.index = index
        self.observerManager = observerManager
        self.mean = mean
        self.sd = sd

        #store an attribute for the pointcloud
        self.numPts = 20 #number of points per bot for depth camera observation
        self._ptcloudData = {} #pointcloud attribute (initialize far away)

    def calc_orientation(self):
        """
        Function to calculate the orientation of the turtlebot WRT the world frame.
        Args:
            None
        Returns:
            Rse: rotation matrix from ego frame to world frame
        """
        qo = (self.observerManager.get_observer_i(self.index)).get_state()
        phi = qo[2, 0]
        Rse = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        return Rse

    def calc_ptcloud(self):
        """
        Function to compute the pointcloud of the environment from the ego turtlebot frame.
        Args:
            None
        Returns:
            ptcloud (3 x N NumPy Array): pointcloud containing (x, y, z) coordinates of obstacles within ego frame.
        """
        #get the ego state vector and orientation
        qEgo = (self.observerManager.get_observer_i(self.index)).get_state()
        pse = np.array([[qEgo[0, 0], qEgo[1, 0], 0]]).T #ego XYZ position
        Rse = self.calc_orientation() #ego rotation matrix to spatial frame

        #get a list of observers for all obstacle turtlebots
        obsIndexList = list(self.observerManager.observerDict.keys()) #get a list of all turtlebot indices
        obsIndexList.remove(self.index) #remove the ego index

        #define an array of angles to generate points over
        thetaArr = np.linspace(0, 2*np.pi, self.numPts).tolist()

        #define an empty pointcloud - should be 3 x numPts
        ptcloud = np.zeros((3, len(obsIndexList) * self.numPts))
        
        #define a number of iterations
        j = 0

        #iterate over all obstacle turtlebots
        for i in obsIndexList:
            #get the state vector and radius of the ith turtlebot
            qo = (self.observerManager.get_observer_i(i)).get_state()
            rt = self.observerManager.dynamics.rTurtlebot

            #calculate the points on the circle - all z values are zero
            xList = [rt*np.cos(theta) + qo[0, 0] for theta in thetaArr]
            yList = [rt*np.sin(theta) + qo[1, 0] for theta in thetaArr]
            zList = [0 for theta in thetaArr]

            #put the points in a numpy array
            ptcloudI = np.array([xList, yList, zList])

            #transform the points into the ego frame
            ptcloudIego = (np.linalg.inv(Rse)@(ptcloudI - pse))

            #store in the ptcloud
            ptcloud[:, j * self.numPts : j*self.numPts + self.numPts] = ptcloudIego

            #increment number of iterations
            j += 1
            
        #store both the pointcloud and the ego state vector at the time the pointcloud is taken
        self._ptcloudData["ptcloud"] = ptcloud
        self._ptcloudData["stateVec"] = qEgo
        return self._ptcloudData
        
    def get_pointcloud(self, update = True):
        """
        Returns the pointcloud dictionary from the class attribute 
        Args:
            update: whether or not to recalculate the pointcloud
        Returns:
            Dict: dictionary of pointcloud points and state vector at time of capture
        """
        #first, calculate the pointcloud
        if update:
            self.calc_ptcloud()
        return self._ptcloudData
    
class DepthCamManager:
    def __init__(self, observerManager, obstacleManager, mean, sd):
        """
        Managerial class to manage the depth cameras for N quadrotors
        Args:
            observerManager (ObserverManager): ObserverManager object instance
            obstacleManager (ObstacleManager): manager for the set of NumObs obstacles
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """

        #store the input parameters
        self.observerManager = observerManager
        self.obstacleManager = obstacleManager
        self.mean = mean
        self.sd = sd

        #create an observer dictionary storing N observer instances
        self.depthCamDict = {}

        #create N lidar objects - one for each turtlebot
        for i in range(self.observerManager.dynamics.N):
            if isinstance(self.observerManager.dynamics, TurtlebotSysDyn):
                #initialize a turtlebot depth camera
                self.depthCamDict[i] = PlanarDepthCamTurtlebot(i, self.observerManager, self.obstacleManager, mean, sd)
            elif isinstance(self.observerManager.dynamics, Qrotor3D):
                #initialize a 3D depth camera
                self.depthCamDict[i] = DepthCam3D(i, self.observerManager, self.obstacleManager, mean, sd)
            else:
                #create a quadrotor depth cam
                self.depthCamDict[i] = PlanarDepthCamQrotor(i, self.observerManager, self.obstacleManager, mean, sd)

    def get_depth_cam_i(self, i):
        """
        Function to retrieve the ith depth camera object for the quadrotor
        Inputs:
            i (integer): index of the quadrotor whose observer we'd like to retrieve
        """

        return self.depthCamDict[i]
    
class LidarManager:
    def __init__(self, observerManager, mean, sd):
        """
        Managerial class to manage the observers for a system of N turtlebots
        Args:
            observerManager (ObserverManager): ObserverManager object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        #store the input parameters
        self.observerManager = observerManager
        self.mean = mean
        self.sd = sd

        #create an observer dictionary storing N observer instances
        self.lidarDict = {}

        #create N lidar objects - one for each turtlebot
        for i in range(self.observerManager.dynamics.N):
            #create an observer with index i
            self.lidarDict[i] = EgoLidar(i, self.observerManager, mean, sd)

    def get_lidar_i(self, i):
        """
        Function to retrieve the ith observer object for the turtlebot
        Inputs:
            i (integer): index of the turtlebot whose observer we'd like to retrieve
        """
        return self.lidarDict[i]