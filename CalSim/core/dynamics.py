import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns

#new for basic function
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons

#for interactive 3d plot
import plotly.graph_objects as go
import k3d
import time  
import imageio
from tempfile import TemporaryDirectory
import os

import yaml


#import utils from transforms.py
from .transforms import *

#import hybrid domain class
from .hybrid_domain import *

class Dynamics:
    """
    Skeleton class for system dynamics
    Includes methods for returning state derivatives, plots, and animations
    """
    def __init__(self, x0, singleStateDimn, singleInputDimn, f, N = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (N*stateDimn x 1 numpy array): (x01, x02, ..., x0N) Initial condition state vector for all N agents
            stateDimn (int): dimension of state vector for a single agent
            inputDimn (int): dimension of input vector for a single agent
            f (python function): dynamics function in xDot = f(x, u, t) -> This is for a SINGLE instance
            N (int): Number of "agents" in the system, i.e. how many instances we want running in parallel
        """    

        #store the dynamics function
        self._f = f

        #store the number of agents
        self.N = N

        #store the state and input dimensions for the extended system
        self.singleStateDimn = singleStateDimn
        self.singleInputDimn = singleInputDimn
        self.sysStateDimn = singleStateDimn * N
        self.sysInputDimn = singleInputDimn * N

        #print("self.singleStateDimn is: ", self.singleStateDimn)
        #print("self.singleInputDimn is: ", self.singleInputDimn)
        #print("self.sysStateDimn is: ", self.sysStateDimn)
        #print("self.sysInputDimn is: ", self.sysInputDimn)

        #store the state and input
        self._x = x0
        self._u = np.zeros((self.sysInputDimn, 1))

        #Store if the system is discrete. Set to be false by default.
        self.is_discrete = False
        self.is_hybrid = False

        #set the plotting style with seaborn
        sns.set_theme()
        sns.set_context("paper")

        with open("<insert path to config>", 'r') as file:
            self.config = yaml.safe_load(file)

    def get_input(self):
        """
        Retrieve the input to the system
        """
        return self._u
    
    def get_state(self):
        """
        Retrieve the state vector
        """

        return self._x
    
    def check_discrete(self):
        """
        Returns true if the system is discrete, false if not
        """

        return self.is_discrete
    
    def check_hybrid(self):
        """
        Returns true if the system is discrete, false if not
        """

        return self.is_hybrid
    
    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """

        print("Skeleton dynamics class")

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """

        return None
        
    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector for the extended system
        Args:
            x (sysStateDimn x 1 numpy array): current state vector at time t
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """

        #assemble the derivative vector for the entire system
        xDot = np.zeros((self.sysStateDimn, 1))
        for i in range(self.N):
            #define slicing indices
            stateSlice0 = self.singleStateDimn * i
            stateSlice1 = self.singleStateDimn * (i + 1)
            inputSlice0 = self.singleInputDimn * i
            inputSlice1 = self.singleInputDimn * (i + 1)

            #extract the state and input of the ith agent
            xi = x[stateSlice0  : stateSlice1, 0].reshape((self.singleStateDimn, 1))
            ui = u[inputSlice0 : inputSlice1, 0].reshape((self.singleInputDimn, 1))
            
            #compute the derivative of the ith agent's state vector
            xDot[stateSlice0 : stateSlice1, 0] = self._f(xi, ui, t).reshape((self.singleStateDimn, ))

        #return the assembled derivative vector
        return xDot
    
    def euler_integrate(self, u, t, dt):
        """
        Integrates system dynamics using Euler integration
        Args:
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (sysStateDimn x 1 numpy array): state vector after integrating
        """

        #integrate starting at x
        self._x = self.get_state() + self.deriv(self.get_state(), u, t)*dt

        #update the input parameter
        self._u = u

        #return integrated state vector
        return self._x
    
    def rk4_integrate(self, u, t, dt):
        """
        Integrates system dynamics using RK4 integration
        Args:
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (sysStateDimn x 1 numpy array): state vector after integrating
        """

        #get current deterministic state
        x = self.get_state()

        #evaluate RK4 constants
        k1 = self.deriv(x, u, t)
        k2 = self.deriv(x + dt*k1/2, u, t + dt/2)
        k3 = self.deriv(x + dt*k2/2, u, t + dt/2)
        k4 = self.deriv(x + dt*k3, u, t + dt)

        #update the state parameter
        self._x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        #update the input parameter
        self._u = u

        #return the integrated state vector
        return self._x
            
    def integrate(self, u, t, dt, integrator = "rk4"):
        """
        Integrate dynamics with either rk4 or euler integration
        Choose either "rk4" or "euler" to select integrator.
        """

        if integrator == "rk4":
            return self.rk4_integrate(u, t, dt)
        else:
            return self.euler_integrate(u, t, dt)
    
    def show_plots_original(self, xData, uData, tData, stateLabels = None, inputLabels = None, obsManager = None):
        """
        Function to show plots specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
            stateLabels ((singleStateDimn) length list of strings): Optional custom labels for the state plots
            inputLabels ((singleInputDimn) length list of strings): Optional custom labels for the input plots
        """

        #get the shapes of the data to plot
        if xData is not None:
            xDataShape = xData.shape[0]//self.N
        else:
            xDataShape = 0
        if uData is not None:
            inputDataShape = uData.shape[0]//self.N
        else:
            inputDataShape = 0

        #Plot each state variable in time
        fig, axs = plt.subplots(xDataShape + inputDataShape)
        fig.suptitle('Evolution of States and Inputs in Time')
        xlabel = 'Time (s)'

        #set state and input labels
        if stateLabels is None:
            stateLabels = ["X" + str(i + 1) for i in range(xDataShape)]
        if inputLabels is None:
            inputLabels = ["U" + str(i + 1) for i in range(inputDataShape)]

        #plot the states for each agent
        for j in range(self.N):
            n = 0 #index in the subplot
            for i in range(xDataShape):
                axs[n].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], xData[xDataShape*j + i, :].tolist()[0:-1])
                axs[n].set(ylabel=stateLabels[n]) #pull labels from the list above
                axs[n].grid()
                n += 1

            #plot the inputs
            for i in range(inputDataShape):
                axs[i+xDataShape].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], uData[inputDataShape*j + i, :].tolist()[0:-1])
                axs[i+xDataShape].set(ylabel=inputLabels[i])
                axs[i+xDataShape].grid()
        
        axs[xDataShape + inputDataShape - 1].set(xlabel = xlabel)
        legendList = ["Agent " + str(i) for i in range(self.N)]
        plt.legend(legendList)
        plt.show()
    
    def show_plots(self, xData, uData, tData, xData1, uData1, tData1, stateLabels = None, inputLabels = None, obsManager = None):
        """
        Function to show plots specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
            stateLabels ((singleStateDimn) length list of strings): Optional custom labels for the state plots
            inputLabels ((singleInputDimn) length list of strings): Optional custom labels for the input plots
        """

        #get the shapes of the data to plot
        if xData is not None:
            xDataShape = xData.shape[0]//self.N
        else:
            xDataShape = 0
        if uData is not None:
            inputDataShape = uData.shape[0]//self.N
        else:
            inputDataShape = 0

        if xData1 is not None:
            xDataShape1 = xData1.shape[0]//self.N
        else:
            xDataShape1 = 0
        if uData1 is not None:
            inputDataShape1 = uData1.shape[0]//self.N
        else:
            inputDataShape1 = 0

        #Plot each state variable in time
        fig, axs = plt.subplots(xDataShape + inputDataShape)
        fig.suptitle('Evolution of States and Inputs in Time')
        xlabel = 'Time (s)'

        #set state and input labels
        if stateLabels is None:
            stateLabels = ["X" + str(i + 1) for i in range(xDataShape)]
        if inputLabels is None:
            inputLabels = ["U" + str(i + 1) for i in range(inputDataShape)]

        """
        #plot the states for each agent
        for j in range(self.N):
            n = 0 #index in the subplot
            for i in range(xDataShape):
                axs[n].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], xData[xDataShape*j + i, :].tolist()[0:-1])
                axs[n].set(ylabel=stateLabels[n]) #pull labels from the list above
                axs[n].grid()
                n += 1

            #plot the inputs
            for i in range(inputDataShape):
                axs[i+xDataShape].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], uData[inputDataShape*j + i, :].tolist()[0:-1])
                axs[i+xDataShape].set(ylabel=inputLabels[i])
                axs[i+xDataShape].grid()
        """        
        n = 0 #index in the subplot
        for i in range(xDataShape):
            axs[n].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], xData[i, :].tolist()[0:-1])
            axs[n].set(ylabel=stateLabels[n]) #pull labels from the list above
            axs[n].grid()
            n += 1

        #plot the inputs
        for i in range(inputDataShape):
            axs[i+xDataShape].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], uData[i, :].tolist()[0:-1])
            axs[i+xDataShape].set(ylabel=inputLabels[i])
            axs[i+xDataShape].grid()

        n = 0
        for i in range(xDataShape1):
            axs[n].plot(tData1.reshape((tData1.shape[1], )).tolist()[0:-1], xData[i, :].tolist()[0:-1])
            axs[n].set(ylabel=stateLabels[n]) #pull labels from the list above
            axs[n].grid()
            n += 1

        #plot the inputs
        for i in range(inputDataShape1):
            axs[i+xDataShape].plot(tData1.reshape((tData1.shape[1], )).tolist()[0:-1], uData[i, :].tolist()[0:-1])
            axs[i+xDataShape].set(ylabel=inputLabels[i])
            axs[i+xDataShape].grid()
        
        axs[xDataShape + inputDataShape - 1].set(xlabel = xlabel)
        legendList = ["Agent " + str(i) for i in range(self.N)]
        plt.legend(legendList)
        plt.show()

    def show_animation(self, xData, uData, tData, axis_lims = None, labels = None, anim_point = None, anim_line = None, animate = False, obsManager = None):
        """
        Function to play animations specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
            axis_lims (Python List, length 4): list of axis limits, ex: [-0.25, 5.25, -0.25, 5.25]
            labels (Python list, length 3): List of strings ["Xlabel", "Ylabel", "Title"] for plot
            anim_point (Python function): function to be called at an index i, returns x, y to be animated
            anim_line (Python function): function to be called at an index i, returns line to be animated
            obsManager (ObstacleManager object): If included, will animate the obstacles in the scene.
        """

        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        
        if animate:
            fig, ax = plt.subplots()
            # set the axes limits
            ax.axis(axis_lims)
            # set equal aspect such that the circle is not shown as ellipse
            ax.set_aspect("equal")

            # create a set of points in the axes
            point, = ax.plot([],[], marker="o", linestyle='None')

            #plot the obstacles if present
            if obsManager is not None:
                #plot the circular obstacles
                for i in range(obsManager.NumObs):
                    #get the obstacle 
                    obsI = obsManager.get_obstacle_i(i)
                    center = obsI.get_center()

                    #plot obstacle on YZ
                    plt.gca().add_patch(plt.Circle((center[1, 0], center[2, 0]), radius = obsI.get_radius(), fc = 'c'))

            #check for anim_line
            if anim_line is not None:
                #define the line for the quadrotor
                line, = ax.plot([], [], 'o-', lw=2)

            def anim_func(i):
                #Call animate_func to get the x and y to plot
                x, y = anim_point(i)

                #set the data to points and return points
                point.set_data(x, y)

                #check for anim_line
                if anim_line is not None:
                    #call the line animation function
                    thisx, thisy = anim_line(i)
                    #set the data for the line
                    line.set_data(thisx, thisy)

                    #if we use a line, return line and point
                    return line, point
                else:
                    #if no line is present, just return the point
                    return point,

            num_frames = xData.shape[1]-1

            anim = animation.FuncAnimation(fig, anim_func, frames=num_frames, interval=1/FREQ*1000, blit=True)

            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(labels[2])
            plt.show()

class Qrotor3D(Dynamics):
    def __init__(self, x0 = np.vstack((np.zeros((3, 1)), np.eye(3).reshape((9, 1)), np.zeros((6, 1)))), m = 0.92, I = 0.0023*np.eye(3), l = 0.15, N = 1):
        """
        Init function for a 3D quadrotor system. State vector is in R18.
        Default state vector has zeros and identity for R.
        State Vector: X = [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33, omegax, omegay, omegaz, xDot, yDot, xDot]
        Input Vector: U = [F, Mx, My, Mz]
        
        Args:
            x0 ((18 x 1) NumPy Array): initial state [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33, omegax, omegay, omegaz, xDot, yDot, xDot]
            m (float): mass of quadrotor in kg
            I ((3x3) NumPy Array): inertia tensor of quadrotor
            l (float): length of one arm of quadrotor
            N (int): number of quadrotors
        """

        #store physical parameters
        self._m = m
        self._I = I
        self._g = 9.81
        self._l = l

        #store a reference to the inverse of the inertia
        self.Iinv = np.linalg.inv(self._I)

        #store geometric parameters
        self.e3 = np.array([[0, 0, 1]]).T

        #define quadrotor dynamics
        def quadrotor_dyn(X, U, t):

            """
            Returns the derivative of the state vector
            Args:
                X (18 x 1 numpy array): current state vector at time t
                U (4 x 1 numpy array): current input vector at time t
                t (float): current time with respect to simulation start
            Returns:
                xDot: state_dimn x 1 derivative of the state vector
            """
            #unpack the input vector
            F, M = U[0, 0], U[1 :, 0].reshape((3, 1))
            F = max(0, F) #Cut off force at zero to prevent negative thrust

            #unpack the state vector
            xDot = X[15:, 0].reshape((3, 1)) #get spatial velocity
            omega = X[12:15, 0].reshape((3, 1)) #angular velocity vector
            omegaHat = hat(omega) #hat map of omega
            R = X[3:12].reshape((3, 3)) #rotation matrix

            #compute the derivatives
            RDot = R @ omegaHat
            omegaDot = self.Iinv @ (M - omegaHat @ self._I @ omega)
            xDDot = F * R @ self.e3 - self._m * self._g * self.e3

            #unpack RDot into a vector
            RDotVec = RDot.reshape((9, 1))
            
            #construct and return state vector        
            return np.vstack((xDot, RDotVec, omegaDot, xDDot))
        
        #call the super init function
        super().__init__(x0, 18, 4, quadrotor_dyn, N = N)

    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """

        print("Planar Quadrotor System")
        print("Mass: ", self._m)
        print("Inertia Tensor: ", self._I)
        print("Gravitational accel: ", self._g)
        print("Arm length: ", self._l)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self._m, self._Ixx, self._g, self._l
        """

        return self._m, self._I, self._g, self._l

    def show_plots(self, xData, uData, tData, xData1, uData1, tData1, obsManager = None):

        #Plot each state variable in time
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Z Vel (m/s)']
        inputLabels = ['Force (N)', '||Moment|| (N*m)']

        #Only plot XYZ position and velocity from the states
        pos = xData[0:3, :]
        vel = xData[15:, :]
        xDataPlot = np.vstack((pos, vel))

        pos1 = xData1[0:3, :]
        vel1 = xData1[15:, :]
        xDataPlot1 = np.vstack((pos1, vel1))

        #Plot the scalar force and magnitude of moment
        fData = uData[0, :].reshape((1, uData.shape[1]))
        mData = np.linalg.norm(uData[1:, :], axis = 0).reshape((1, uData.shape[1]))
        uDataPlot = np.vstack((fData, mData))

        fData1 = uData1[0, :].reshape((1, uData1.shape[1]))
        mData1 = np.linalg.norm(uData1[1:, :], axis = 0).reshape((1, uData1.shape[1]))
        uDataPlot1 = np.vstack((fData1, mData1))

        super().show_plots(xDataPlot, uDataPlot, tData, xDataPlot1, uDataPlot1, tData1, stateLabels, inputLabels)

        #Now, plot the 3D trajectory and obstacles
        x = pos[0, :].tolist()
        y = pos[1, :].tolist()
        z = pos[2, :].tolist()

        x1 = pos1[0, :].tolist()
        y1 = pos1[1, :].tolist()
        z1 = pos1[2, :].tolist()

        ax = plt.figure().add_subplot(projection='3d')
        #ax1 = plt.figure().add_subplot(projection='3d')

        print("x is", x)
        print("y is", y)
        print("z is", z)
        print("size x is: ", len(x))

        print("x1 is", x1)
        print("y1 is", y1)
        print("z1 is", z1)
        print("size x1 is: ", len(x1))

        ax.plot(x, y, z, color='r')
        ax.plot(x1, y1, z1, color='g')
        
        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                # # #get a grid of points
                # u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                # x1 = center[0, 0] + radius * np.cos(u) * np.sin(v)
                # y1 = center[1, 0] + radius * np.sin(u) * np.sin(v)
                # z1 = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)
        
        #set axis limits
        ax.set_xlim3d([-0.5, 2])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.5, 2])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, 2.5])
        ax.set_zlabel('Z')

        #plot the 3d trajectory
        plt.title("Quadrotor Trajectory")
        plt.show()

    def show_interactive_k3d_full(self, xHist, xHist1, obsManager=None, save_path='animation.mp4'):
        """
        Visualizes the dynamic 3D movement of two agents (robots) over time, with static obstacles, using k3d.
        Saves frames and compiles them into a video if a save path is provided.
        
        Args:
            xHist (ndarray): State history of agent 1, Nx3 array where N is the number of frames.
            xHist1 (ndarray): State history of agent 2, Nx3 array where N is the number of frames.
            obsManager (ObstacleManager, optional): Manager holding information about obstacles.
            save_path (str): Path to save the animation video.
        """
        # Create a k3d plot
        plot = k3d.plot()

        # Prepare agent trajectories in Nx3 format
        agent1_trajectory = np.vstack((xHist[0, :], xHist[1, :], xHist[2, :])).T
        agent2_trajectory = np.vstack((xHist1[0, :], xHist1[1, :], xHist1[2, :])).T

        # Initialize lines for agent trajectories
        agent1_line = k3d.line(agent1_trajectory[:1].astype(np.float32), shader='flat', color=0x0000FF, width=0.1)
        agent2_line = k3d.line(agent2_trajectory[:1].astype(np.float32), shader='flat', color=0xFF0000, width=0.1)
        plot += agent1_line
        plot += agent2_line

        # Plot obstacles from obsManager, if available
        if obsManager is not None:
            for i in range(obsManager.NumObs):
                obs = obsManager.get_obstacle_i(i)
                center = obs.get_center().flatten()
                radius = obs.get_radius()

                # Generate obstacle as a spherical mesh
                phi, theta = np.mgrid[0:np.pi:20j, 0:2 * np.pi:30j]
                x = radius * np.sin(phi) * np.cos(theta) + center[0]
                y = radius * np.sin(phi) * np.sin(theta) + center[1]
                z = radius * np.cos(phi) + center[2]
                vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
                indices = []

                for j in range(19):
                    for k in range(29):
                        p1 = j * 30 + k
                        p2 = p1 + 1
                        p3 = p1 + 30
                        p4 = p3 + 1
                        indices.extend([p1, p2, p3, p2, p3, p4])

                obstacle_mesh = k3d.mesh(vertices.astype(np.float32), indices=np.array(indices, dtype=np.uint32), color=0x00FF00, opacity=0.3)
                plot += obstacle_mesh

        # Animation parameters
        frames = []
        with TemporaryDirectory() as temp_dir:
            for i in range(len(agent1_trajectory)):
                # Update line data for each frame
                agent1_line.vertices = agent1_trajectory[:i+1].astype(np.float32)
                agent2_line.vertices = agent2_trajectory[:i+1].astype(np.float32)

                # Capture the frame
                img_data = plot.fetch_snapshot()
                print(f"Frame {i}: IMG DATA IS: ", img_data)

                if img_data:
                    frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    with open(frame_path, 'wb') as f:
                        f.write(img_data)
                    frames.append(imageio.imread(frame_path))
                else:
                    print(f"Warning: No valid snapshot fetched for frame {i}")

                # Brief pause for animation effect
                time.sleep(0.05)  # Adjust for desired speed

        # Save as mp4 video if frames were successfully captured
        if frames and save_path:
            imageio.mimsave(save_path, frames, fps=20)
            print(f"Animation saved at {save_path}")

        # Display interactive plot
        plot.display()


    def show_interactive_3d_animation_k3d(self, xHist, xHist1, obsManager=None):
        """
        Visualizes the 3D movement of two agents (robots) and any static obstacles using k3d.
        Args:
            xHist (ndarray): State history of agent 1, Nx3 array where N is the number of frames.
            xHist1 (ndarray): State history of agent 2, Nx3 array where N is the number of frames.
            obsManager (ObstacleManager, optional): Manager holding information about obstacles.
        """
        # Create a k3d plot
        plot = k3d.plot()

        # Verify that xHist and xHist1 are in the shape (N, 3)
        agent1_trajectory = np.vstack((xHist[0, :], xHist[1, :], xHist[2, :])).T
        agent2_trajectory = np.vstack((xHist1[0, :], xHist1[1, :], xHist1[2, :])).T

        # Plot agent 1 trajectory line with wider width
        agent1_line = k3d.line(agent1_trajectory.astype(np.float32), shader='flat', color=0x0000FF, width=0.1)
        plot += agent1_line

        # Plot agent 2 trajectory line with wider width
        agent2_line = k3d.line(agent2_trajectory.astype(np.float32), shader='flat', color=0xFF0000, width=0.1)
        plot += agent2_line

        # Plot obstacles from obsManager, if available
        if obsManager is not None:
            for i in range(obsManager.NumObs):
                obs = obsManager.get_obstacle_i(i)
                center = obs.get_center().flatten()
                radius = obs.get_radius()

                # Generate a spherical mesh for the obstacle
                phi, theta = np.mgrid[0:np.pi:20j, 0:2 * np.pi:30j]
                x = radius * np.sin(phi) * np.cos(theta) + center[0]
                y = radius * np.sin(phi) * np.sin(theta) + center[1]
                z = radius * np.cos(phi) + center[2]

                # Flatten arrays for k3d mesh
                vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
                indices = []

                for j in range(19):
                    for k in range(29):
                        p1 = j * 30 + k
                        p2 = p1 + 1
                        p3 = p1 + 30
                        p4 = p3 + 1
                        indices.extend([p1, p2, p3, p2, p3, p4])

                obstacle_mesh = k3d.mesh(vertices.astype(np.float32), indices=np.array(indices, dtype=np.uint32), color=0x00FF00, opacity=0.3)
                plot += obstacle_mesh

        # Set up camera and plot properties
        plot.camera_auto_fit = False
        plot.camera = [5.0, 5.0, 5.0, 0, 0, 0, 0, 0, 1]  # Adjusted camera for better view
        plot.grid_visible = True
        plot.axes_helper = 5

        # Display the plot
        plot.display()


    def show_interactive_3d_animation_plotly(self, xHist, xHist1, obsManager=None):
        """
        Visualizes the 3D movement of two agents (robots) and any static obstacles using Plotly.
        Args:
            xHist (ndarray): State history of agent 1, Nx3 array where N is the number of frames.
            xHist1 (ndarray): State history of agent 2, Nx3 array where N is the number of frames.
            obsManager (ObstacleManager, optional): Manager holding information about obstacles.
        """
        fig = go.Figure()

        # Plot agent 1 trajectory
        fig.add_trace(go.Scatter3d(
            x=xHist[0, :], y=xHist[1, :], z=xHist[2, :],
            mode='lines+markers', name='Agent 1', line=dict(color='blue')
        ))

        # Plot agent 2 trajectory
        fig.add_trace(go.Scatter3d(
            x=xHist1[0, :], y=xHist1[1, :], z=xHist1[2, :],
            mode='lines+markers', name='Agent 2', line=dict(color='red')
        ))

        # Plot obstacles from obsManager, if available
        if obsManager is not None:
            for i in range(obsManager.NumObs):
                obs = obsManager.get_obstacle_i(i)
                center = obs.get_center()
                radius = obs.get_radius()

                # Generate points for a spherical surface
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                # Add each obstacle as a 3D surface
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z, opacity=0.3, colorscale='Viridis',
                    showscale=False, name=f'Obstacle {i+1}'
                ))

        # Set up plot limits, labels, and title
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[-5, 5]),
                yaxis=dict(title='Y', range=[-5, 5]),
                zaxis=dict(title='Z', range=[0, 5])
            ),
            title="3D Interactive Animation with Obstacles and Robots",
            showlegend=True
        )

        # Show interactive plot
        fig.show()


    def show_full_animation(self, xHist, xHist1, pointcloud1, pointcloud2, obsManager=None):
        """
        Visualizes the simulation of the agents and their point clouds with optional toggles for visibility.
        Inputs:
            xHist: Full state history of agent 1 (Nx3 array where N is the number of frames)
            xHist1: Full state history of agent 2 (Nx3 array)
            pointcloud1: List of point cloud data arrays for agent 1, with each item a 3xM array (M points)
            pointcloud2: List of point cloud data arrays for agent 2
            obsManager: Obstacle manager (optional) to show static obstacles
        """
        # Extract only position data (x, y, z) for animation
        xHist_pos = xHist[:3, :]
        xHist1_pos = xHist1[:3, :]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles if present
        if obsManager is not None:
            for i in range(obsManager.NumObs):
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()

                # Generate a sphere grid for obstacles
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                # Plot the obstacle surfaces
                #ax.plot_surface(x, y, z, color="grey", alpha=0.3)
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)

        # Initialize agent plots and point clouds
        robot1_plot, = ax.plot([], [], [], 'bo', label="Agent 1")
        robot2_plot, = ax.plot([], [], [], 'ro', label="Agent 2")
        pointcloud1_plot = ax.scatter([], [], [], c='blue', marker='x', label="Agent 1 Point Cloud")
        pointcloud2_plot = ax.scatter([], [], [], c='red', marker='x', label="Agent 2 Point Cloud")

        # Toggle visibility for agents and point clouds
        visibility = [True, True, True, True]
        rax = plt.axes([0.05, 0.4, 0.15, 0.2], facecolor='lightgoldenrodyellow')
        check = CheckButtons(rax, ['Robot 1', 'Robot 2', 'Pointcloud 1', 'Pointcloud 2'], visibility)

        def toggle_visibility(label):
            index = ['Robot 1', 'Robot 2', 'Pointcloud 1', 'Pointcloud 2'].index(label)
            visibility[index] = not visibility[index]

            robot1_plot.set_visible(visibility[0])
            robot2_plot.set_visible(visibility[1])
            pointcloud1_plot.set_visible(visibility[2])
            pointcloud2_plot.set_visible(visibility[3])
            plt.draw()

        check.on_clicked(toggle_visibility)

        # Update function for the animation
        def update(frame):
            # Ensure data at current frame is valid
            if frame < xHist_pos.shape[1] and frame < xHist1_pos.shape[1]:
                # Update agent 1's position
                robot1_plot.set_data([xHist_pos[0, frame]], [xHist_pos[1, frame]])
                robot1_plot.set_3d_properties([xHist_pos[2, frame]])

                # Update agent 2's position
                robot2_plot.set_data([xHist1_pos[0, frame]], [xHist1_pos[1, frame]])
                robot2_plot.set_3d_properties([xHist1_pos[2, frame]])

                # Update point cloud for agent 1 if pointcloud1 has data for this frame
                if frame < len(pointcloud1):
                    pc1 = pointcloud1[frame]
                    pointcloud1_plot._offsets3d = (pc1[0, :], pc1[1, :], pc1[2, :])

                # Update point cloud for agent 2 if pointcloud2 has data for this frame
                if frame < len(pointcloud2):
                    pc2 = pointcloud2[frame]
                    pointcloud2_plot._offsets3d = (pc2[0, :], pc2[1, :], pc2[2, :])

                return robot1_plot, robot2_plot, pointcloud1_plot, pointcloud2_plot

        # Set plot limits and labels
        ax.set_xlim3d([-0.5, 2])
        ax.set_ylim3d([0, 4])
        ax.set_zlim3d([0, 2.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("3D Animation with Point Clouds")

        # Set up animation and save to file
        anim = FuncAnimation(fig, update, frames=min(xHist_pos.shape[1], len(pointcloud1)), interval=50, blit=False)
        anim.save('10_31_show_pcworld_1.mp4')

        plt.show()


    def show_animation_basic(self, xData, xData1, obsManager = None):

        FREQ = 30
        pos = xData[0:3, :]
        pos1 = xData1[0:3, :]

        x1 = pos[0, :].tolist()
        y1 = pos[1, :].tolist()
        z1 = pos[2, :].tolist()

        x2 = pos1[0, :].tolist()
        y2 = pos1[1, :].tolist()
        z2 = pos1[2, :].tolist()
        
        # Set up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                # # #get a grid of points
                # u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                # x1 = center[0, 0] + radius * np.cos(u) * np.sin(v)
                # y1 = center[1, 0] + radius * np.sin(u) * np.sin(v)
                # z1 = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)
        

        # Set the viewing limits for the 3D plot
        #ax.set_xlim(min(x1 + x2) - 1, max(x1 + x2) + 1)
        #ax.set_ylim(min(y1 + y2) - 1, max(y1 + y2) + 1)
        #ax.set_zlim(min(z1 + z2) - 1, max(z1 + z2) + 1)

        ax.set_xlim3d([-0.5, 2])
        ax.set_ylim3d([0, 4])
        ax.set_zlim3d([0, 2.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Plot the initial positions of the robots
        robot1_line, = ax.plot([], [], [], 'ro-', label="Robot 1")  # 'ro-' means red circles with a line
        robot2_line, = ax.plot([], [], [], 'bo-', label="Robot 2")  # 'bo-' means blue circles with a line

        # Set labels
        
        ax.legend()

        # Initialize the function for the animation
        def init():
            robot1_line.set_data([], [])
            robot1_line.set_3d_properties([])

            robot2_line.set_data([], [])
            robot2_line.set_3d_properties([])

            return robot1_line, robot2_line

        # Animation function to update the positions
        def update(frame):
            # For robot 1
            robot1_line.set_data(x1[:frame], y1[:frame])
            robot1_line.set_3d_properties(z1[:frame])
            
            # For robot 2
            robot2_line.set_data(x2[:frame], y2[:frame])
            robot2_line.set_3d_properties(z2[:frame])
            
            return robot1_line, robot2_line

        # Create the animation
        frames = len(x1)  # The number of frames is equal to the number of position points
        anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1/FREQ*1000)
        anim.save('11_04_show_animation_3.mp4')

        # Show the plot
        plt.show()

    def show_animation(self, xData, uData, tData, xData1, uData1, tData1, animate = True, obsManager = None):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
            obsManager (ObstacleManager): Manager object for obstacles. If included, will animate the obstacles.
        """

        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        L = self._l #quadrotor arm length

        #initialize figure and a point
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.axis('square')

        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)

        #define points reprenting the center and propellers of the quadrotor
        x, y, z = [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        points, = ax.plot(x, y, z, 'o', color = 'r')

        #x1, y1, z1 = [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        #points1, = ax.plot(x1, y1, z1, 'o', color = 'g')

        #define lines representing the arms of the quadrotor
        lines = [ax.plot([], [], [])[0] for _ in range(2)]
        #lines1 = [ax.plot([], [], [])[0] for _ in range(2)]

        def update(num, data, lines):

            #get the rotation matrix and position at num
            R = data[3:12, num].reshape((3, 3))
            p = data[0:3, num].reshape((3, 1))

            #define points on arm1 and arm2 in the quadrotor frame
            pArm11 = np.array([[0, -L, 0]]).T #first point on arm 1
            pArm12 = np.array([[0, L, 0]]).T #second point on arm 1
            pArm21 = np.array([[L, 0, 0]]).T #first point on arm 1
            pArm22 = np.array([[-L, 0, 0]]).T #second point on arm 1

            #transform the points into the spatial frame
            pArm11S = R @ pArm11 + p
            pArm12S = R @ pArm12 + p
            pArm21S = R @ pArm21 + p
            pArm22S = R @ pArm22 + p

            #define a line between the points
            lines[0].set_data(np.hstack((pArm11S, pArm12S))[0:2, :])
            lines[0].set_3d_properties(np.hstack((pArm11S, pArm12S))[2, :])
            lines[1].set_data(np.hstack((pArm21S, pArm22S))[0:2, :])
            lines[1].set_3d_properties(np.hstack((pArm21S, pArm22S))[2, :])

            #define the x points to plot
            xPoints = [data[0, num], pArm11S[0, 0],  pArm12S[0, 0],  pArm22S[0, 0],  pArm21S[0, 0]]
            yPoints = [data[1, num], pArm11S[1, 0],  pArm12S[1, 0],  pArm22S[1, 0],  pArm21S[1, 0]]
            zPoints = [data[2, num], pArm11S[2, 0],  pArm12S[2, 0],  pArm22S[2, 0],  pArm21S[2, 0]]

            #define a point
            points.set_data(xPoints, yPoints)
            points.set_3d_properties(zPoints, 'z')

        # Setting the axes properties
        ax.set_xlim3d([-0.5, 2])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.5, 2])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, 2.5])
        ax.set_zlabel('Z')

        #run animation
        num_frames = xData.shape[1]-1
        print("num frames in animation is: ", num_frames)
        #anim = animation.FuncAnimation(fig, update,  num_frames, fargs=(xData, xData1), interval=1/FREQ*1000, blit=False)
        anim = animation.FuncAnimation(fig, update,  num_frames, fargs=(xData, lines), interval=1/FREQ*1000, blit=False)
        anim1 = animation.FuncAnimation(fig, update,  num_frames, fargs=(xData1, lines), interval=1/FREQ*1000, blit=False)

        anim.save('animation.mp4')
        anim1.save('animation1.mp4')
        plt.show()

    def show_animation_original(self, xData, uData, tData, animate = True, obsManager = None):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
            obsManager (ObstacleManager): Manager object for obstacles. If included, will animate the obstacles.
        """
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        L = self._l #quadrotor arm length

        #initialize figure and a point
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.axis('square')

        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)

        #define points reprenting the center and propellers of the quadrotor
        x, y, z = [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        points, = ax.plot(x, y, z, 'o')

        #define lines representing the arms of the quadrotor
        lines = [ax.plot([], [], [])[0] for _ in range(2)]

        def update(num, data, lines):
            #get the rotation matrix and position at num
            R = data[3:12, num].reshape((3, 3))
            p = data[0:3, num].reshape((3, 1))

            #define points on arm1 and arm2 in the quadrotor frame
            pArm11 = np.array([[0, -L, 0]]).T #first point on arm 1
            pArm12 = np.array([[0, L, 0]]).T #second point on arm 1
            pArm21 = np.array([[L, 0, 0]]).T #first point on arm 1
            pArm22 = np.array([[-L, 0, 0]]).T #second point on arm 1

            #transform the points into the spatial frame
            pArm11S = R @ pArm11 + p
            pArm12S = R @ pArm12 + p
            pArm21S = R @ pArm21 + p
            pArm22S = R @ pArm22 + p

            #define a line between the points
            lines[0].set_data(np.hstack((pArm11S, pArm12S))[0:2, :])
            lines[0].set_3d_properties(np.hstack((pArm11S, pArm12S))[2, :])
            lines[1].set_data(np.hstack((pArm21S, pArm22S))[0:2, :])
            lines[1].set_3d_properties(np.hstack((pArm21S, pArm22S))[2, :])

            #define the x points to plot
            xPoints = [data[0, num], pArm11S[0, 0],  pArm12S[0, 0],  pArm22S[0, 0],  pArm21S[0, 0]]
            yPoints = [data[1, num], pArm11S[1, 0],  pArm12S[1, 0],  pArm22S[1, 0],  pArm21S[1, 0]]
            zPoints = [data[2, num], pArm11S[2, 0],  pArm12S[2, 0],  pArm22S[2, 0],  pArm21S[2, 0]]

            #define a point
            points.set_data(xPoints, yPoints)
            points.set_3d_properties(zPoints, 'z')

        # Setting the axes properties
        ax.set_xlim3d([-0.5, 2])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.5, 2])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, 2.5])
        ax.set_zlabel('Z')

        #run animation
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig, update,  num_frames, fargs=(xData, lines), interval=1/FREQ*1000, blit=False)
        anim.save
        plt.show()

    def show_plots_original(self, xData, uData, tData, obsManager = None):
        #Plot each state variable in time
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Z Vel (m/s)']
        inputLabels = ['Force (N)', '||Moment|| (N*m)']

        #Only plot XYZ position and velocity from the states
        pos = xData[0:3, :]
        vel = xData[15:, :]
        xDataPlot = np.vstack((pos, vel))

        #Plot the scalar force and magnitude of moment
        fData = uData[0, :].reshape((1, uData.shape[1]))
        mData = np.linalg.norm(uData[1:, :], axis = 0).reshape((1, uData.shape[1]))
        uDataPlot = np.vstack((fData, mData))

        super().show_plots(xDataPlot, uDataPlot, tData, stateLabels, inputLabels)

        #Now, plot the 3D trajectory and obstacles
        x = pos[0, :].tolist()
        y = pos[1, :].tolist()
        z = pos[2, :].tolist()
        ax = plt.figure().add_subplot(projection='3d')

        ax.plot(x, y, z)
        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)
        
        #set axis limits
        ax.set_xlim3d([-0.5, 2])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.5, 2])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, 2.5])
        ax.set_zlabel('Z')

        #plot the 3d trajectory
        plt.title("Quadrotor Trajectory")
        plt.show()


class DiscreteDynamics(Dynamics):
    """
    Init function for a discrete time dynamics class.
    Has the form x(k+1) = f(x, u, k)
    """
    def __init__(self, x0, singleStateDimn, singleInputDimn, f, N = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (N*stateDimn x 1 numpy array): (x01, x02, ..., x0N) Initial condition state vector for all N agents
            stateDimn (int): dimension of state vector for a single agent
            inputDimn (int): dimension of input vector for a single agent
            f (python function): dynamics function in xDot = f(x, u, t) -> This is for a SINGLE instance
            N (int): Number of "agents" in the system, i.e. how many instances we want running in parallel
        """    
        super().__init__(x0, singleStateDimn, singleInputDimn, f, N)

        #override the discrete parameter
        self.is_discrete = True

    def integrate(self, u, t, dt, integrator = None):
        """
        Step the dynamics forward using a discrete time step. Here, the ``dt"
        argument is ignored, and the function f is all that is used to step to x(k+1).
        """
        #get current deterministic state
        x = self.get_state()

        #update the stored deterministic state and input
        self._x = self.deriv(x, u, t)
        self._u = u

        #return the next time step state
        return self._x
    

class HybridDynamics(Dynamics):
    """
    Hybrid Dynamics Class
    """
    def __init__(self, x0, singleStateDimn, singleInputDimn, domainList, N = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (N*stateDimn x 1 numpy array): (x01, x02, ..., x0N) Initial condition state vector for all N agents
            stateDimn (int): dimension of state vector for a single agent
            inputDimn (int): dimension of input vector for a single agent
            f (python function): dynamics function in xDot = f(x, u, t) -> This is for a SINGLE instance
            N (int): Number of "agents" in the system, i.e. how many instances we want running in parallel
        """
        #call the super init function with NONE for f
        super().__init__(x0, singleStateDimn, singleInputDimn, None, N)

        #override the hybrid parameter
        self.is_hybrid = True

        #Extract the indices of the different subsystems
        indexList = [Di.get_index() for Di in domainList]

        #Store the domains in a dictionary indexed by the domain index
        self.domainDict = {k:v for (k, v) in zip(indexList, domainList)}

        #store the current domain -> initialize as the first domain in the index list
        self.Di = self.domainDict[indexList[0]]

    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector for the extended system
        Args:
            x (sysStateDimn x 1 numpy array): current state vector at time t
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        #assemble the derivative vector for the entire system
        xDot = np.zeros((self.sysStateDimn, 1))
        for i in range(self.N):
            #define slicing indices
            stateSlice0 = self.singleStateDimn * i
            stateSlice1 = self.singleStateDimn * (i + 1)
            inputSlice0 = self.singleInputDimn * i
            inputSlice1 = self.singleInputDimn * (i + 1)

            #extract the state and input of the ith agent
            xi = x[stateSlice0  : stateSlice1, 0].reshape((self.singleStateDimn, 1))
            ui = u[inputSlice0 : inputSlice1, 0].reshape((self.singleInputDimn, 1))
            
            #compute the derivative of the ith agent's state vector -> NOTE: USES the current domain's dynaics
            xDot[stateSlice0 : stateSlice1, 0] = self.Di.f(xi, ui, t).reshape((self.singleStateDimn, ))

        #return the assembled derivative vector
        return xDot

    def integrate(self, u, t, dt, integrator = 'rk4'):
        """
        Step the dynamics forward using a discrete time step. Here, the ``dt"
        argument is ignored, and the function f is all that is used to step to x(k+1).
        """
        #get current deterministic state
        x = self.get_state()

        #check for domain switch using the guards
        for i in range(len(self.Di.SList)):
            #extract the guard function from the SList of the current domain
            s = self.Di.SList[i]

            #Check if the guard condition is met
            if s(x, u, t):
                #Update state and index of new domain -> This should be the INTERNAL domain index
                self._x, newInd = self.Di.DeltaList[i](x, u, t)

                #update the stored input
                self._u = u

                #Switch to new domain domain using the new index -> call the corresonding domain from the dict
                self.Di = self.domainDict[newInd]

                #return the new post-hybrid-transition state
                return self._x

        #update the stored deterministic state and input
        if integrator == "rk4":
            return self.rk4_integrate(u, t, dt)
        else:
            return self.euler_integrate(u, t, dt)
    
"""
**********************************
PLACE YOUR DYNAMICS FUNCTIONS HERE
**********************************
"""

class DoubleIntegratorDyn(Dynamics):
    """
    Double Integrator System
    May initialize N integrators to run in parallel.
    """
    def __init__(self, x0, N = 1):
        #define the double integrator dynamics
        def double_integrator(x, u, t):
            """
            Double integrator dynamics
                x (2x1 NumPy array): state vector
                u (1x1 NumPy array): input vector
                t (float): current time
            """
            return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ u
        super().__init__(x0, 2, 1, double_integrator, N = N)
    
    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Double Integrator Dynamics")

class SimpleDiscrete(DiscreteDynamics):
    """
    Simple discrete dynamics, x(k+1) = x(k) + u(k)
    """
    def __init__(self, x0, N = 1):
        #define the dynamics
        def f(x, u, t):
            """
            Simple discrete time dynamics
                x (1x1 NumPy array): state vector
                u (1x1 NumPy array): input vector
                t (float): current time
            """
            return x + u
        super().__init__(x0, 1, 1, f, N = N)
    
    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Simple Discrete Dynamics")

class MSDRamp(Dynamics):
    """
    Mass-spring-damper system on a ramp.
    """
    def __init__(self, x0, m = 0.5, g = 9.81, k = 15, b = 0.5, theta = np.pi/6, N = 1):
        """
        Inputs:
            x0 (2x1 NumPy Array): Initial condition
            m (float): mass in kg
            g (float): acceleration due to gravity (m/s^2)
            k (float): spring constant (N/m)
            b (float): damping constant (N/(m/s))
            theta (float): angle of ramp
            N (int): number of agents
        """
        self.m = m
        self.g = g
        self.k = k
        self.b = b
        self.theta = theta

        def msd_ramp(x, u, t):
            """
            Mass spring damper ramp dynamics
            Inputs:
                x (2x1 NumPy array): current state vector
                u (1x1 NumPy Array): force applied to mass (typically 0)
                t (float): current time in simulation
            """
            return np.array([[x[1, 0]], [u[0, 0]/m -k/m * x[0, 0] - b/m * x[1, 0] - g*np.sin(theta)]])
        
        #call the init function on the MSD ramp system
        super().__init__(x0, 2, 1, msd_ramp, N = N)

    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Mass-Spring-Damper System")
        print("Mass: ", self.m)
        print("Gravitational accel: ", self.g)
        print("Spring constant: ", self.k)
        print("Damping constant: ", self.b)
        print("Ramp angle: ", self.theta)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self.m, self.g, self.k, self.b, self.theta
        """
        return self.m, self.g, self.k, self.b, self.theta

class DoublePendulum(Dynamics):
    """
    Double pendulum dynamics
    """
    def __init__(self, x0, m = 0.5, I = 0.5, L = 1, g = 9.81, N = 1):
        """
        Inputs:
            x0: initial condition [q1, q2, q1Dot, q2Dot]
            m: mass of joints
            I: inertia of joints
            L: length of joints
            g: gravitational acceleration
        """
        #store the system parameters
        self.m = m
        self.I = I
        self.L = L
        self.g = g

        def double_pend(x, u, t):
            """
            Double pendulum dynamics
            Includes a torque input at each joint.
            """
            #extract the states
            q1, q2, q1Dot, q2Dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

            #compute the inertia matrix
            M = np.array([[2*I + 1.5*m*L**2 + m*L**2*np.cos(q2), I+0.25*m*L**2+0.5*m*L**2*np.cos(q2)],
                          [I+0.25*m*L**2+0.5*m*L**2*np.cos(q2), I+0.25*m*L**2]])
            #compute C
            C = np.array([[0, -m*L**2*q1Dot*np.sin(q2)-0.5*m*L**2*q2Dot*np.sin(q2)], [0.5*m*L**2*q1Dot*np.sin(q2), 0]])

            #compute N
            N = np.array([[0.5*m*g*L*(np.sin(q1+q2)+3*np.sin(q1)), 0.5*m*g*L*np.sin(q1+q2)]]).T

            #compute qDDot
            qDot = np.array([[q1Dot, q2Dot]]).T
            qDDot = np.linalg.inv(M)@(u - C@qDot - N)

            #return [qDot, qDDot]
            return np.vstack((qDot, qDDot))
        
        #call the init function on the double pendulum system
        super().__init__(x0, 4, 2, double_pend, N = N)
    
    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Double Pendulum System")
        print("Mass: ", self.m)
        print("Inertia: ", self.I)
        print("Length: ", self.L)
        print("Gravitational accel: ", self.g)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self.m, self.I, self.L, self.g
        """
        return self.m, self.I, self.L, self.g

class FlywheelPendulum(Dynamics):
    """
    Pendulum driven by a flywheel
    """

class MarsProbe(Dynamics):
    """
    Dynamics of a particle falling through the atmosphere of mars.
    """
    def __init__(self, x0, m = 572.743, Cd = 1.46, A = 5.5155, ftMax = 9806.65, N = 1):
        """
        Init function for mars probe.
        
        Mission Phases and Probe Parameters:
        - https://ntrs.nasa.gov/api/citations/20080034645/downloads/20080034645.pdf
        - Enter atmosphere at 5600 m/s
        - Chute deployment occurs at Mach 1.65, 12.9 km above surface
        - Touchdown occurs at 0.7 m/s
        
        Inputs:
            x0 (2x1 NumPy Array): position vector of the system
            m (float): mass of the probe
            Cd (float): drag coefficient of probe 
                NOTE: Cd for a parachute ~0.5 -> add this to Cd for just probe. Chute area ~200m^2.
                      https://downloads.spj.sciencemag.org/space/2022/9805457.pdf
            A (float): surface area for drag calculation
            ftMax (float): maximum thrust force of probe
            N (int): number of probes to simulate
        """
        #properties of probe
        self.m = m
        self.Cd = Cd
        self.ftMax = ftMax

        #properties of Mars
        self.rho = 0.02 #density of atmosphere (kg/m^3)
        self.M = 6.41693 * 10**23 #mass of Mars (kg)
        self.R = 3390 * 10**3 #radius of Mars (m)
        self.G = 6.6743 * 10**(-11) #universal gravitational constant

        def probe_dyn(x, u, t):
            """
            Mars probe dynamics.
            Inputs:
                x (2x1 NumPy Array): [y, yDot] for y distance to the center of mars
                u (1x1 NumPy Array): thrust force of the probe
                t (float): time
            """
            y = x[0, 0]
            yDot = x[1, 0]
            yDDot = -self.G * self.M / (y**2) + 0.5*self.rho*self.Cd*yDot**2/self.m + u[0, 0]/m
            return np.array([[yDot, yDDot]]).T
        
        #call the init function on the probe pendulum system
        super().__init__(x0, 2, 1, probe_dyn, N = N)

class TurtlebotSysDyn(Dynamics):
    """
    System of N Turtlebots
    """
    def __init__(self, x0, N = 1, rTurtlebot = 0.15):
        """
        Init function for a system of N turtlebots.
        Args:
            x0 (NumPy Array): (x1, y1, phi1, ..., xN, yN, phiN) initial condition for all N turtlebots
            N (Int, optional): number of turtlebots in the system
            rTurtlebot (float): radius of the turtlebots in the system
        """

        #define the turtlebot dynamics
        def f_turtlebot(x, u, t):
            #extract the orientation angle of the Nth turtlebot
            PHI = x[2, 0]
            return np.array([[np.cos(PHI), 0], [np.sin(PHI), 0], [0, 1]])@u

        #call the super init function to create a turtlebot system
        super().__init__(x0, 3, 2, f_turtlebot, N)

        #store a copy of the augmented input vector for feedback linearization
        self._z = np.zeros((self.sysInputDimn, 1))

        #store the turtlebot radius
        self.rTurtlebot = rTurtlebot 

    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Turtlebot System")
        print("Turtlebot radius: ", self.rTurtlebot)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self.rTurtlebot
        """
        return self.rTurtlebot

    def set_z(self, z, i):
        """
        Function to set the value of z, the augmented input vctor.
        Inputs:
            z ((2N x 1) NumPy Array): Augmented input vector
            i (int): index we wish to place the updated z at
        """
        #store in class attribute
        self._z[2*i : 2*i + 2, 0] = z.reshape((2, ))
    
    def get_z(self):
        """
        Function to return the augmented input vector, z, at any point.
        """
        #retrieve and return augmented input vector
        return self._z
    
    def show_plots(self, xData, uData, tData, obsManager = None):
        #Plot the spatial trajectory of the turtlebots
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        #iterate over each turtlebot state vector
        for j in range(self.N):
            xCoords = xData[3*j, :].tolist() #extract all of the velocity data to plot on the y axis
            yCoords = xData[3*j+1, :].tolist() #remove the last point, get artefacting for some reason
            ax.plot(xCoords[0:-1], yCoords[0:-1])
        
        legendList = ["Agent " + str(i) for i in range(self.N)]
        plt.legend(legendList)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Positions of Turtlebots in Space")
        plt.show()

        #call the super plots with custom labels to show the individual states
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Phi (rad)']
        inputLabels = ["V (m/s)", "Omega (rad/s)"]
        super().show_plots(xData, uData, tData, stateLabels, inputLabels)
    
    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
            obsManager (ObstacleManager): if included, will animate any obstacles present
        """
        #define animation function to return the x, y points to be animated
        def anim_point(i):
            x = []
            y = []
            #get the x, y data associated with each turtlebot
            for j in range(self.N):
                x.append(xData[3*j, i])
                y.append(xData[3*j+1, i])
            return x, y
    
        #define axis limits
        axis_lims = [-0.25, 5.25, -0.25, 5.25]

        #define labels
        axis_labels = ["X Position (m)", "Y Position (m)", "Positions of Turtlebots in Space"]

        #call the super() animation function
        super().show_animation(xData, uData, tData, axis_lims, axis_labels, anim_point, anim_line = None, animate = animate, obsManager = obsManager)



class PlanarQrotor(Dynamics):
    def __init__(self, x0 = np.zeros((8, 1)), m = 0.92, Ixx = 0.0023, l = 0.15, N = 1):
        """
        Init function for a Planar quadrotor system.
        State Vector: X = [x, y, z, theta, x_dot, y_dot, z_dot, theta_dot]
        Input Vector: U = [F, M]
        
        Args:
            x0 ((8 x 1) NumPy Array): initial state (x, y, z, theta, x_dot, y_dot, z_dot, theta_dot)
            m (float): mass of quadrotor in kg
            Ixx (float): moment of inertia about x axis of quadrotor
            l (float): length of one arm of quadrotor
            N (int): number of quadrotors
        """
        #store physical parameters
        self._m = m
        self._Ixx = Ixx
        self._g = 9.81
        self._l = l

        #define quadrotor dynamics
        def quadrotor_dyn(X, U, t):
            """
            Returns the derivative of the state vector
            Args:
                X (8 x 1 numpy array): current state vector at time t
                U (2 x 1 numpy array): current input vector at time t
                t (float): current time with respect to simulation start
            Returns:
                xDot: state_dimn x 1 derivative of the state vector
            """
            #unpack the input vector
            F, M = U[0, 0], U[1, 0]
            F = max(0, F) #Cut off force at zero to prevent negative thrust
            
            #unpack the state vector
            x_dot, y_dot, z_dot = X[4, 0], X[5, 0], X[6, 0] #velocities
            theta, theta_dot = X[3, 0], X[7, 0] #orientations
            
            #calculate the second time derivatives of each
            x_ddot = 0
            y_ddot = (-np.sin(theta)*F)/self._m
            z_ddot = (np.cos(theta)*F - self._m*self._g)/self._m
            theta_ddot = M/self._Ixx
            
            #construct and return state vector        
            return np.array([[x_dot, y_dot, z_dot, theta_dot, x_ddot, y_ddot, z_ddot, theta_ddot]]).T
        
        #call the super init function
        super().__init__(x0, 8, 2, quadrotor_dyn, N = N)

    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        print("Planar Quadrotor System")
        print("Mass: ", self._m)
        print("Inertia: ", self._Ixx)
        print("Gravitational accel: ", self._g)
        print("Arm length: ", self._l)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self._m, self._Ixx, self._g, self._l
        """
        return self._m, self._Ixx, self._g, self._l

    def show_plots(self, xData, uData, tData, obsManager = None):
        #Plot each state variable in time
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'Theta (rad)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Z Vel (m/s)', 'Angular Vel (rad/s)']
        inputLabels = ['Force (N)', 'Moment (N*m)']
        super().show_plots(xData, uData, tData, stateLabels, inputLabels)
        
    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
            obsManager (ObstacleManager): Manager object for obstacles. If included, will animate the obstacles.
        """
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        L = self._l #quadrotor arm length

        #define point animation function
        def anim_point(i):
            y = xData[1, i]
            z = xData[2, i]
            return y, z
        
        #define line animation function
        def anim_line(i):
            y = xData[1, i]
            z = xData[2, i]

            #draw the quadrotor line body
            theta = xData[3, i]
            x1 = y + L*np.cos(theta)
            x2 = y - L*np.cos(theta)
            y1 = z + L*np.sin(theta)
            y2 = z - L*np.sin(theta)
            thisx = [x1, x2]
            thisy = [y1, y2]
            return thisx, thisy
        
        #define axis limits
        axis_lims = [-1, 2.5, 0, 2.5]

        #define plot labels
        labels = ["Y Position (m)", "Z Position (m)", "Position of Drone in Space"]

        #call the super animation function
        super().show_animation(xData, uData, tData, axis_lims, labels, anim_point, anim_line, animate = animate, obsManager=obsManager)

class TiltRotor(Dynamics):
    """
    Class for tilt rotor drone based on rotation matrix dynamics
    """
    def __init__(self, x0 = np.vstack((np.zeros((3, 1)), np.eye(3).reshape((9, 1)), np.zeros((6, 1)))), m = 0.92, I = 0.0023*np.eye(3), l = 0.15, N = 1):
        """
        Init function for a 3D tiltrotor system. State vector is in R18.
        Default state vector has zeros and identity for R.
        State Vector: X = [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33, omegax, omegay, omegaz, xDot, yDot, xDot]
        Input Vector: U = [T1, T2, T3, T4, Beta1, Beta2, Beta3, Beta4]
        
        Args:
            x0 ((18 x 1) NumPy Array): initial state [x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33, omegax, omegay, omegaz, xDot, yDot, xDot]
            m (float): mass of quadrotor in kg
            I ((3x3) NumPy Array): inertia tensor of quadrotor
            l (float): length of one arm of quadrotor
            N (int): number of quadrotors
        """
        #store physical parameters
        self._m = m
        self._I = I
        self._g = 9.81
        self._l = l

        #store a reference to the inverse of the inertia
        self.Iinv = np.linalg.inv(self._I)

        #store geometric parameters
        self.e3 = np.array([[0, 0, 1]]).T

        #calculate the ri vectors assuming a square body
        r1 = np.array([[l/np.sqrt(2), l/np.sqrt(2), 0]]).T
        r2 = np.array([[-l/np.sqrt(2), l/np.sqrt(2), 0]]).T
        r3 = np.array([[-l/np.sqrt(2), -l/np.sqrt(2), 0]]).T
        r4 = np.array([[l/np.sqrt(2), -l/np.sqrt(2), 0]]).T

        def calcABeta(beta):
            """
            Helper function to compute A(beta) matrix
            This matrix maps from thrust vector to total force
            """
            #compute the rodrigues matrices
            Rr1 = rodrigues(r1/np.linalg.norm(r1), beta[0, 0])
            Rr2 = rodrigues(r2/np.linalg.norm(r2), beta[1, 0])
            Rr3 = rodrigues(r3/np.linalg.norm(r3), beta[2, 0])
            Rr4 = rodrigues(r4/np.linalg.norm(r4), beta[3, 0])

            #form the matrix
            return np.hstack((Rr1 @ self.e3, Rr2 @ self.e3, Rr3 @ self.e3, Rr4 @ self.e3))

        def calcBBeta(beta):
            """
            Helper function to compute B matrix
            """
            #compute rHat matrix
            rHat = np.hstack((hat_3d(r1), hat_3d(r2), hat_3d(r3), hat_3d(r4)))

            #compute the rodrigues matrices
            Rr1 = rodrigues(r1/np.linalg.norm(r1), beta[0, 0])
            Rr2 = rodrigues(r2/np.linalg.norm(r2), beta[1, 0])
            Rr3 = rodrigues(r3/np.linalg.norm(r3), beta[2, 0])
            Rr4 = rodrigues(r4/np.linalg.norm(r4), beta[3, 0])

            #compute rotation matrix term
            row1 = np.hstack((Rr1 @ self.e3, np.zeros((3, 3))))
            row2 = np.hstack((np.zeros((3, 1)), Rr2 @ self.e3, np.zeros((3, 2))))
            row3 = np.hstack((np.zeros((3, 2)), Rr3 @ self.e3, np.zeros((3, 1))))
            row4 = np.hstack((np.zeros((3, 3)), Rr4 @ self.e3))
            Rterm = np.vstack((row1, row2, row3, row4))

            #return the product
            return rHat @ Rterm

        #define quadrotor dynamics
        def quadrotor_dyn(X, U, t):
            """
            Returns the derivative of the state vector
            Args:
                X (18 x 1 numpy array): current state vector at time t
                U (8 x 1 numpy array): current input vector at time t
                t (float): current time with respect to simulation start
            Returns:
                xDot: state_dimn x 1 derivative of the state vector
            """
            #unpack the input vector into thrusts and angles
            T, BETA = U[0:4, :].reshape((4, 1)), U[4:, ].reshape((4, 1))

            #Compute force and moment using the linear maps
            F = calcABeta(BETA) @ T
            M = calcBBeta(BETA) @ T

            #unpack the state vector
            xDot = X[15:, 0].reshape((3, 1)) #get spatial velocity
            omega = X[12:15, 0].reshape((3, 1)) #angular velocity vector
            omegaHat = hat(omega) #hat map of omega
            R = X[3:12].reshape((3, 3)) #rotation matrix

            #compute the derivatives
            RDot = R @ omegaHat
            omegaDot = self.Iinv @ (M - omegaHat @ self._I @ omega)
            xDDot = R @ F - self._m * self._g * self.e3

            #unpack RDot into a vector
            RDotVec = RDot.reshape((9, 1))
            
            #construct and return state vector        
            return np.vstack((xDot, RDotVec, omegaDot, xDDot))
        
        #call the super init function
        super().__init__(x0, 18, 8, quadrotor_dyn, N = N)

    def print_params(self):
        """
        Prints the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        """
        Qrotor3D.print_params(self)

    def return_params(self):
        """
        Returns the parameters of the dynamics object.
        e.g. mass, gravitational acceleration, ...
        Returns:
            self._m, self._Ixx, self._g, self._l
        """
        return Qrotor3D.return_params(self)
    
    def show_plots(self, xData, uData, tData, obsManager = None):
        #Plot each state variable in time
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Z Vel (m/s)']
        inputLabels = ['T1', 'T2', 'T3', 'T4', 'B1', 'B2', 'B3', 'B4']

        #Only plot XYZ position and velocity from the states
        pos = xData[0:3, :]
        vel = xData[15:, :]
        xDataPlot = np.vstack((pos, vel))
        super().show_plots(xDataPlot, None, tData, stateLabels, None)

        #now, call the input data plot
        super().show_plots(None, uData, tData, None, inputLabels)

        #Now, plot the 3D trajectory and obstacles
        x = pos[0, :].tolist()
        y = pos[1, :].tolist()
        z = pos[2, :].tolist()
        ax = plt.figure().add_subplot(projection='3d')

        ax.plot(x, y, z)
        #plot the obstacles if present
        if obsManager is not None:
            #plot the circular obstacles
            for i in range(obsManager.NumObs):
                #get the obstacle 
                obsI = obsManager.get_obstacle_i(i)
                center = obsI.get_center()
                radius = obsI.get_radius()
                
                #get a grid of points
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = center[0, 0] + radius * np.cos(u) * np.sin(v)
                y = center[1, 0] + radius * np.sin(u) * np.sin(v)
                z = center[2, 0] + radius * np.cos(v)

                #plot the 3D surface on the axes
                ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0)
        
        #set axis limits
        ax.set_xlim3d([-0.5, 2])
        ax.set_xlabel('X')
        ax.set_ylim3d([-0.5, 2])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, 2.5])
        ax.set_zlabel('Z')

        #plot the 3d trajectory
        plt.title("Quadrotor Trajectory")
        plt.show()

    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        return Qrotor3D.show_animation(self, xData, uData, tData, animate, obsManager)

class BouncingBall(HybridDynamics):
    """
    Bouncing ball hybrid system
    """
    def __init__(self, x0, N=1):
        """
        Init function for a bouncing ball
        """
        #define spatial parameters
        self.g = 9.81 #graviational acceleration
        self.e = 0.9 #coefficient of restitution

        #define tolerance for guard condition
        self.guardTol = 0.001

        #Define continuous dynamics function
        def f0(x, u, t):
            """
            Dynamics function for bouncing ball
            Inputs:
                x (2x1 NumPy Array): State Vector
                u (0x0): Input Vector (not used)
                t (float): time
            """
            return np.array([[x[1, 0], -self.g]]).T
        
        #Define guard for state 0
        def s0(x, u, t):
            """
            Guard function for continuous phase of bouncing ball
            NOTE: We use a tolerance on the exact condition here!
            Inputs:
                x (2x1 NumPy Array): State vector
                u (Unused)
                t (Unused)
            Returns:
                True or False: Guard condition met or not
            """
            if abs(x[0, 0]) <= self.guardTol and x[1, 0] <= 0:
                return True
            else:
                return False
        
        #Define transition map for state 0
        def delta0(x, u, t):
            """
            Transition map for delta 1. Returns to state 0.
            Inputs:
                x (2x1 NumPy Array): State vector
                u (unused)
                t (unused)
            Returns:
                x+ (2x1 NumPy Array): state after transition
                nextIndex (Integer): index after transition
            """
            #flip velocity, multiply by coeff of restitution, and return to DOMAIN 0
            return np.array([[x[0, 0], -self.e * x[1, 0]]]).T, 0

        #Create a Hybrid Domain object
        domainList = [HybridDomain(f0, [s0], [delta0], 0)]

        #Call the Super init function
        super().__init__(x0, singleStateDimn = 2, singleInputDimn = 0, domainList = domainList, N = 1)

class SimpleTwoDomainHybrid(HybridDynamics):
    """
    Simple Two-Domain Hybrid System
    """
    def __init__(self, x0, N=1):
        """
        Init function for a bouncing ball
        """
        #define spatial parameters
        self.g = 9.81 #graviational acceleration
        self.e = 0.7 #0.9 #coefficient of restitution

        #define tolerance for guard condition
        self.guardTol = 0.004

        #Define continuous dynamics function
        def f0(x, u, t):
            """
            Dynamics function for accelerating down phase
            Inputs:
                x (2x1 NumPy Array): State Vector
                u (0x0): Input Vector (not used)
                t (float): time
            """
            return np.array([[x[1, 0], -self.g]]).T
        
        def f1(x, u, t):
            """
            Dynamics function for accelerating up phase
            Inputs:
                x (2x1 NumPy Array): State Vector
                u (0x0): Input Vector (not used)
                t (float): time
            """
            return np.array([[x[1, 0], self.g]]).T
        
        #Define guard for state 0
        def s0(x, u, t):
            """
            Guard function for continuous phase of bouncing ball
            NOTE: We use a tolerance on the exact condition here!
            Inputs:
                x (2x1 NumPy Array): State vector
                u (Unused)
                t (Unused)
            Returns:
                True or False: Guard condition met or not
            """
            if abs(x[0, 0]) <= self.guardTol and x[1, 0] <= 0:
                return True
            else:
                return False
            
        #Define guard for state 1
        def s1(x, u, t):
            """
            Guard function for continuous phase of bouncing ball
            NOTE: We use a tolerance on the exact condition here!
            Inputs:
                x (2x1 NumPy Array): State vector
                u (Unused)
                t (Unused)
            Returns:
                True or False: Guard condition met or not
            """
            if abs(x[0, 0] - 1) <= self.guardTol and x[1, 0] >= 0:
                return True
            else:
                return False
        
        #Define transition map for state 0
        def delta0(x, u, t):
            """
            Transition map for delta 1. Returns to state 0.
            Inputs:
                x (2x1 NumPy Array): State vector
                u (unused)
                t (unused)
            Returns:
                x+ (2x1 NumPy Array): state after transition
                nextIndex (Integer): index after transition
            """
            #flip velocity, multiply by coeff of restitution, and return to DOMAIN 0
            return np.array([[x[0, 0], -self.e * x[1, 0]]]).T, 1
        
        def delta1(x, u, t):
            """
            Transition map for delta 1. Returns to state 0.
            Inputs:
                x (2x1 NumPy Array): State vector
                u (unused)
                t (unused)
            Returns:
                x+ (2x1 NumPy Array): state after transition
                nextIndex (Integer): index after transition
            """
            #flip velocity, multiply by coeff of restitution, and return to DOMAIN 0
            return np.array([[x[0, 0], -self.e * x[1, 0]]]).T, 0

        #Create a Hybrid Domain object
        domainList = [HybridDomain(f0, [s0], [delta0], 0), HybridDomain(f1, [s1], [delta1], 1)]

        #Call the Super init function
        super().__init__(x0, singleStateDimn = 2, singleInputDimn = 0, domainList = domainList, N = 1)