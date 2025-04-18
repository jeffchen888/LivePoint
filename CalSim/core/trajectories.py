import numpy as np

class Trajectory:
    def __init__(self, start, end, T):
        """
        Init function for linear tracking trajectories in RN.
        Generates a smooth straight line trajectory with zero start and end velocity. Uses sinusoidal interpolation.
        Args:
            start (Nx1 numpy array): initial spatial position in N dimensions (NOT initial state vector)
            end (Nx1 numpy array): final spatial position in N dimensions
            T (float): trajectory period
        """

        self.x0 = start
        self.xF = end
        self.spatialDimn = self.x0.shape[0]
        self.T = T

    def pos(self, t):
        """
        Function to get desired position at time t
        Args:
            t (float): current time
        Returns:
            (Nx1 numpy array): position coordinates for the quadrotor to track at time t
        """
        #use sinusoidal interpolation to get a smooth trajectory with zero velocity at endpoints
        if t>self.T:
            #if beyond the time of the trajectory end, return the desired position as a setpoint
            return self.xF
        des_pos = (self.xF-self.x0)/2*np.sin(t*np.pi/self.T - np.pi/2)+(self.x0+self.xF)/2
        return des_pos #calculates all three at once
    
    def vel(self, t):
        """
        Function to get the desired velocity at time t
        Inputs:
            t: current time
        Returns:
            (Nx1 Numpy array): velocity for the system to track at time t
        """

        #differentiate position
        if t>self.T:
            #If beyond the time of the trajectory end, return 0 as desired velocity
            return np.zeros((self.spatialDimn, 1))
        des_vel = (self.xF-self.x0)/2*np.cos(t*np.pi/self.T - np.pi/2)*np.pi/self.T
        return des_vel

    def accel(self, t):
        """
        Function to get the desired acceleration at time t
        Args:
            t: current time
        Returns:
            (Nx1 Numpy array): acceleration for the system to track at time t
        """

        #differentiate acceleration
        if t>self.T:
            #If beyond the time of the trajectory end, return 0 as desired acceleration
            return np.zeros((self.spatialDimn, 1))
        des_accel = -(self.xF-self.x0)/2*np.sin(t*np.pi/self.T - np.pi/2)*(np.pi/self.T)**2
        return des_accel

    def get_state(self, t):
        """
        Function to get the desired position, velocity, and accel at a time t
        Inputs:
            t: current time
        Returns:
            x_d, v_d, a_d: desired position, velocity, and acceleration at time t
        """

        return self.pos(t), self.vel(t), self.accel(t)
    
class TrajectoryManager:
    def __init__(self, starts, ends, Ts, spatialDimn = 8, N = 1):
        """
        Managerial trajectory class for a system of N turtlebots - creates N trajectory objects
        Inputs:
            starts (NumPy Array): vStacked start positions (NumPy arrays) for each turtlebot
            ends (NumPy Array): vStacked goal positions (NumPy arrays) for each turtlebot
            Ts (float): list of desired time for each trajectory
            N: number of turtlebots in the system
        """

        self.trajDict = {}
        self.spatialDimn = spatialDimn #single spatial dimension

        for i in range(N):
            #create a trajectory
            startI = starts[self.spatialDimn*i : self.spatialDimn*i + self.spatialDimn].reshape((self.spatialDimn, 1))
            endI = ends[self.spatialDimn*i : self.spatialDimn*i+self.spatialDimn].reshape((self.spatialDimn, 1))
            self.trajDict[i] = Trajectory(startI, endI, Ts)

    def get_traj_i(self, i):
        """
        Function to retrieve the ith trajectory object in the system
        Inputs:
            i (integer): index of trajectory we want to return
        """

        return self.trajDict[i]