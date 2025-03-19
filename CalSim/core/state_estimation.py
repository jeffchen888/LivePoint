import numpy as np
from .dynamics import *

class StateObserver:
    def __init__(self, dynamics, mean = None, sd = None, index = None):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
            index (int): index of the agent
        """

        self.dynamics = dynamics
        self.mean = mean
        self.sd = sd
        self.index = index

        #store the state dimension of an individual agent
        self.singleStateDimn = dynamics.singleStateDimn
        self.singleInputDimn = dynamics.singleInputDimn

        #store the state dimension of an entire agent
        self.sysStateDimn = dynamics.sysStateDimn
        self.sysInputDimn = dynamics.sysInputDimn
        
    def get_state(self, return_full = False):
        """
        Returns a potentially noisy observation of the system state
        Inputs:
            return_full (Boolean): return the entire state vector of the system instead of just index i
        """

        #first, get an observation of the entire state vector
        if self.mean or self.sd:
            #get an observation of the vector with noise
            observedState = self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.sysStateDimn, 1))
        else:
            #get an observation of the vector with no noise
            observedState = self.dynamics.get_state()

        #now, return either the entire observed system state vector or the observed state vector of a single agent
        if return_full:
            #return the entire state vector
            return observedState
        else:
            #return the state vector of the agent at index self.index
            #print("self.singleStateDimn*self.index is: ", self.singleStateDimn*self.index)
            #print("self.singleStateDimn*(self.index + 1) is: ", self.singleStateDimn*(self.index + 1))
            return observedState[self.singleStateDimn*self.index : self.singleStateDimn*(self.index + 1)].reshape((self.singleStateDimn, 1))
    
class TurtlebotObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for a state observer for a single agent within a system of N agents
        Args:
            dynamics (Dynamics): Dynamics object for the entir system
            mean (float): Mean for gaussian noise. Defaults to None.
            sd (float): standard deviation for gaussian noise. Defaults to None.
            index (int): index of the agent in the system
        """
        #initialize the super class
        super().__init__(dynamics, mean, sd, index)

        #store the index of the agent
        self.index = index

        #store the state dimension of an individual agent
        self.singleStateDimn = dynamics.singleStateDimn
        self.singleInputDimn = dynamics.singleInputDimn
    
    def get_state(self):
        """
        Returns a potentially noisy measurement of the state vector of the ith turtlebot
        Returns:
            (Dynamics.singleStateDimn x 1 NumPy array), observed state vector of the ith turtlebot in the system (zero indexed)
        """
        return super().get_state(return_full=True)[self.singleStateDimn*self.index : self.singleStateDimn*(self.index + 1)].reshape((self.singleStateDimn, 1))
    
    def get_pos(self):
        """
        Returns the XYZ position of the turtlebot. Note that z is always zero.
        """
        #get the x, y, theta state vector
        qFull = self.get_state()

        #return x, y, z
        return np.vstack(([qFull[0:2].reshape((2, 1)), 0]))
    
    def get_orient(self):
        """
        Returns the orientation angle phi of the turtlebot
        """
        return self.get_state()[2, 0]

    def get_vel(self):
        """
        Returns a potentially noisy measurement of the derivative of the state vector of the ith agent
        Returns:
            (Dynamics.singleStateDimn x 1 NumPy array): observed derivative of the state vector of the ith turtlebot in the system (zero indexed)
        """
        #first, get the current input to the system
        u = self.dynamics.get_input()

        #now, get the noisy measurement of the entire state vector
        x = super().get_state(return_full = True)

        #get the full velocity vector of the system
        vel = self.dynamics.deriv(x, u, 0) 
        
        #return the correct slice of thefull velocity vector
        return vel[self.singleStateDimn*self.index : self.singleStateDimn*(self.index + 1)].reshape((self.singleStateDimn, 1))

class QuadObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for state observer for a planar quadrotor

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
            index (int): index of the quadrotor
        """
        super().__init__(dynamics, mean, sd, index)
    
    def get_pos(self):
        """
        Returns a potentially noisy measurement of JUST the position of the Qrotor mass center
        Returns:
            3x1 numpy array, observed position vector of system
        """
        return self.get_state()[0:3].reshape((3, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of JUST the spatial velocity of the Qrotor mass center
        Returns:
            3x1 numpy array, observed velocity vector of system
        """
        return self.get_state()[4:7].reshape((3, 1))

    def get_orient(self):
        """
        Returns a potentially noisy measurement of the 
        Assumes that the system is planar and just rotates about the X axis.
        Returns:
            theta (float): orientation angle of quadrotor with respect to world frame
        """
        return self.get_state()[3, 0]
    
    def get_omega(self):
        """
        Returns a potentially noisy measurement of the angular velocity theta dot
        Assumes that the system is planar and just rotates about the X axis.
        Returns:
            theta (float): orientation angle of quadrotor with respect to world frame
        """
        return self.get_state()[7, 0]
    

class Quad3DObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for state observer for a 3D quadrotor

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
            index (int): index of the quadrotor
        """

        super().__init__(dynamics, mean, sd, index)
    
    def get_pos(self):
        """
        Returns a potentially noisy measurement of JUST the position of the Qrotor mass center
        Returns:
            3x1 numpy array, observed position vector of system
        """

        return self.get_state()[0:3].reshape((3, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of JUST the spatial velocity of the Qrotor mass center
        Returns:
            3x1 numpy array, observed spatial velocity vector of system
        """

        return self.get_state()[15:].reshape((3, 1))

    def get_orient(self):
        """
        Returns a potentially noisy measurement of the 
        Assumes that the system is planar and just rotates about the X axis.
        Returns:
            theta (float): orientation angle of quadrotor with respect to world frame
        """

        return self.get_state()[3:12].reshape((3, 3))
    
    def get_omega(self):
        """
        Returns a potentially noisy measurement of the angular velocity theta dot
        Assumes that the system is planar and just rotates about the X axis.
        Returns:
            theta (float): orientation angle of quadrotor with respect to world frame
        """

        return self.get_state()[12:15].reshape((3, 1))

class ObserverManager:
    def __init__(self, dynamics, mean = None, sd = None, ObserverClass = None):
        """
        Managerial class to manage the observers for a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
            ObserverClass (StateObserver Class): Custom Observer Class, Inherits from StateObserver
        """

        #store the input parameters
        self.dynamics = dynamics
        self.mean = mean
        self.sd = sd

        #create an observer dictionary storing N observer instances
        self.observerDict = {}

        #create N observer objects of the correct type
        for i in range(self.dynamics.N):
            #create an observer with index i. Check the type of dynamics object and use a custom observer where necessary.
            if isinstance(self.dynamics, TurtlebotSysDyn):
                #create a turtlebot observer
                self.observerDict[i] = TurtlebotObserver(dynamics, mean, sd, i)
            elif isinstance(self.dynamics, PlanarQrotor):
                #create a planar quadrotor observer
                self.observerDict[i] = QuadObserver(dynamics, mean, sd, i)
            elif isinstance(self.dynamics, Qrotor3D) or isinstance(self.dynamics, TiltRotor):
                #create a 3D quadrotor observer for a 3d quad or tiltrotor
                self.observerDict[i] = Quad3DObserver(dynamics, mean, sd, i)
            elif ObserverClass is not None:
                self.observerDict[i] = ObserverClass(dynamics, mean, sd, i)
            else:
                #create a standard state observer
                self.observerDict[i] = StateObserver(dynamics, mean, sd, i)

    def get_observer_i(self, i):
        """
        Function to retrieve the ith observer object for the turtlebot
        Inputs:
            i (integet): index of the turtlebot whose observer we'd like to retrieve
        """

        return self.observerDict[i]
    
    def get_state(self):
        """
        Returns a potentially noisy observation of the *entire* system state (vector for all N bots)
        """

        #get each individual observer state
        xHatList = []
        for i in range(self.dynamics.N):
            #call get state from the ith observer
            #print("self.get_observer_i(i) is: ", self.get_observer_i(i))
            #print("self.get_observer_i(i).get_state() is: ", self.get_observer_i(i).get_state())
            xHatList.append(self.get_observer_i(i).get_state())

        #vstack the individual observer states
        return np.vstack(xHatList)