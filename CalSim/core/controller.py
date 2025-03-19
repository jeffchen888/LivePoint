import numpy as np
from .state_estimation import *
import yaml

"""
File containing controllers 
"""  
class Controller:
    def __init__(self, observer, lyapunovBarrierList = None, trajectory = None, depthCam = None):
        """
        Skeleton class for feedback controllers
        Args:
            observer (Observer): state observer object
            lyapunov (List of LyapunovBarrier): list of LyapunovBarrier objects
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            depthCam (PlanarDepthCam or Lidar): depth camera object
        """
        #store input parameters
        self.observer = observer
        self.lyapunovBarrierList = lyapunovBarrierList
        self.trajectory = trajectory
        self.depthCam = depthCam
        
        #store input
        self._u = np.zeros((self.observer.singleInputDimn, 1))
    
    def eval_input(self, t):
        """
        Solve for and return control input. This function is to be implemented by the user.

        (This is a placeholder method meant to be overridden by specific controllers to compute input based on t)
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.singleInputDimn x 1)): input vector, as determined by controller
        """

        return np.zeros((self.observer.singleInputDimn, 1))
    
    def set_input(self, t):
        """
        Sets the input parameter at time t by calling eval_input
        Inputs:
            u ((Dynamics.singleInputDimn x 1)): input vector
        """

        self._u = self.eval_input(t)
        return self._u

    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """

        return self._u

class ControllerManager(Controller):
    def __init__(self, observerManager, ControlType, barrierManager = None, trajectoryManager = None, depthManager = None, delta = 0.1):
        """
        Managerial class that points to N controller instances for the system. Interfaces
        directly with the overall system dynamics object.
        Args:
            observerManager (ObserverManager)
            barrierManager (BarrierManager)
            trajectoryManager (TrajectoryManager)
            depthManager (LidarManager or DepthManager)
            ControlType (Class Name): Name of a class for a controller
        """

        #store input parameters
        self.observerManager = observerManager
        self.barrierManager = barrierManager
        self.trajectoryManager = trajectoryManager
        self.depthManager = depthManager
        self.ControlType = ControlType
        with open("<insert path to config>", 'r') as file:
            self.config = yaml.safe_load(file)

        #store the input parameter (should not be called directly but with get_input)
        self._u = None

        #get the number of agents in the system
        self.N = self.observerManager.dynamics.N

        #create a controller dictionary
        self.controllerDict = {}

        #create N separate controllers - one for each agent
        for i in range(self.N):

            #extract the ith trajectory
            try:
                trajI = self.trajectoryManager.get_traj_i(i)
            except:
                print("No trajectory passed in.")
                trajI = None

            #get the ith observer object
            try:
                egoObsvI = self.observerManager.get_observer_i(i)
            except:
                print("No state observer passed in.")
                egoObsvI = None

            #get the ith barrier object
            try:
                barrierI = self.barrierManager.get_barrier_list_i(i)
            except:
                print("No barrier/lyapunov object passed in.")

                barrierI = None

            #get the ith depth camera object
            try:
                depthI = self.depthManager.get_depth_cam_i(i)
            except:
                print("No depth camera/LIDAR object passed in.")
                depthI = None

            #create a controller of the specified type
            self.controllerDict[i] = self.ControlType(egoObsvI, barrierI, trajI, depthI, delta)

    def eval_input(self, t):
        """
        Solve for and return control input for all N agents. Solves for the input ui to 
        each agent in the system and assembles all of the input vectors into a large 
        input vector for the entire system.
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.sysInputDimn x 1)): input vector, as determined by controller
        """

        #initilialize input vector as zero vector - only want to store once all have been updated
        u = np.zeros((self.observerManager.dynamics.sysInputDimn, 1))
        singleInputDimn = self.observerManager.dynamics.singleInputDimn

        #loop over the system to find the input to each agent
        for i in range(self.N):
            #solve for the latest input to agent i, store the input in the u vector
            self.controllerDict[i].set_input(t)
            u[singleInputDimn*i : singleInputDimn*(i + 1)] = self.controllerDict[i].get_input()

        #store the u vector in self._u
        self._u = u

        #return the full input vector
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        

        return self._u
    

class FFController(Controller):
    """
    Class for a simple feedforward controller.
    """
    def __init__(self, observer, lyapunovBarrierList = None, trajectory = None, depthCam = None):
        """
        Init function for a ff controller
        Args:
            observer (Observer): state observer object
            ff (NumPy Array): constant feedforward input to send to system
            lyapunov (List of LyapunovBarrier): list of LyapunovBarrier objects
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
        """
        self.ff = np.array([[9.81*1, 0]]).T
        # self.ff = np.vstack((np.ones((4, 1)) * 9.81 * 0.92 * 0.25, np.array([[0, 0, 0, 0]]).T)) #tiltrotor FF
        # self.ff = np.array([[1]])
        self.depthCam = depthCam
        super().__init__(observer, lyapunovBarrierList, trajectory)

    def eval_input(self, t):
        """
        Solve for and return control input
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.singleInputDimn x 1)): input vector, as determined by controller
        """
        # print("called")
        if self.depthCam is not None:
            #test the depth camera
            self.depthCam.get_pointcloud()
        return self.ff