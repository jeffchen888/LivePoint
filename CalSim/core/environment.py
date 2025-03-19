import numpy as np
from .controller import ControllerManager, Controller
from .state_estimation import ObserverManager
from .obstacle import ObstacleManager
from .depth_cam import DepthCamManager
from .dynamics import Dynamics, DiscreteDynamics, Qrotor3D
from .force_controllers import QRotorGeometricPD
from .trajectories import *
from .orca import Vector, ORCAAgent

import sys
import yaml
from os import path

class Environment:
    def __init__(self, dynamics, dynamics1, controller = None, controller1 = None,
                 observer = None, observer1 = None, obstacleManager = None,
                 obstacleManager1 = None, orig_obstacleManager = None,
                 trajManager = None, trajManager1 = None, depthManager = None, depthManager1 = None, T = 10):
        """
        Initializes a simulation environment
        Args:
            dynamics (Dynamics): system dynamics object
            controller (Controller): system controller object
            observer (Observer): system state estimation object
            obstacleManager (ObstacleManager): obstacle objects
            T (Float): simulation time
        """

        #store system parameters
        self.dynamics = dynamics #store system dynamics
        self.dynamics1 = dynamics1
        self.obstacleManager = obstacleManager
        self.obstacleManager1 = obstacleManager1 #store manager for any obstacles present
        self.originalObstacleManager = orig_obstacleManager
        self.trajManager = trajManager
        self.trajManager1 = trajManager1
        self.depthManager = depthManager
        self.depthManager1 = depthManager1
        self.T = T

        self.pointcloudHistory = []
        self.pointcloudHistory1 = []
        self.pointcloudworldHistory = []
        self.pointcloudworldHistory1 = []

        #if observer and controller are none, create default objects
        if observer is not None:
            self.observer = observer
        else:
            #create a default noise-free observer manager
            self.observer = ObserverManager(dynamics)
        if observer1 is not None:
            self.observer1 = observer1
        else:
            #create a default noise-free observer manager
            self.observer1 = ObserverManager(dynamics1)
        
        if controller is not None:
            self.controller = controller
        else:
            #create a default zero input controller manager (using the skeleton Controller class)
            self.controller = ControllerManager(self.observer, Controller, None, None, None)
        if controller1 is not None:
            self.controller1 = controller1
        else:
            #create a default zero input controller manager (using the skeleton Controller class)
            self.controller1 = ControllerManager(self.observer1, Controller, None, None, None)
        
        with open("<insert path to config>", 'r') as file:
            self.config = yaml.safe_load(file)
        
        #define environment parameters
        self.iter = 0 #number of iterations
        self.iter1 = 0
        self.t = 0 #time in seconds 
        self.done = False
        self.step_number = 0

        #Store system state
        self.x = self.dynamics.get_state() #Actual state of the system
        self.x0 = self.x #store initial condition for use in reset
        self.xObsv = None #state as read by the observer
        self.ptCloud = None #point cloud state as read by vision
        
        #Store system1 state
        self.x1 = self.dynamics1.get_state() #Actual state of the system
        self.x01 = self.x1 #store initial condition for use in reset
        self.xObsv1 = None #state as read by the observer
        self.ptCloud1 = None #point cloud state as read by vision

        self.setsingle = False
        #Define simulation parameters
        if not self.dynamics.check_discrete():
            print("dynamics are not discrete (in environment.py)")
            #If the dynamics are not discrete, set frequency params. as cont. time

            self.SIM_FREQ = 1000 #integration frequency in Hz
            self.CONTROL_FREQ = 50 #control frequency in Hz

            # self.SIM_FREQ = 1000 #integration frequency in Hz
            # self.CONTROL_FREQ = 50 #control frequency in Hz
            self.SIMS_PER_STEP = self.SIM_FREQ//self.CONTROL_FREQ
            self.TOTAL_SIM_TIME = T #total simulation time in s
            self.TOTAL_ITER = self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1 #total number of iterations
        else:
            print("dynamics are discrete (in environment.py)")
            #if dynamics are discrete, set the frequency to be 1 (1 time step)
            self.SIM_FREQ = 1
            self.CONTROL_FREQ = 1
            self.SIMS_PER_STEP = 1
            self.TOTAL_SIM_TIME = T #T now represents total sime time
            self.TOTAL_ITER = self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1 #total number of iterations

        #just for testing
        self.TOTAL_ITER = 400

        print("self.TOTAL_ITER is: (in environment.py): ", self.TOTAL_ITER)
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.sysStateDimn, self.TOTAL_ITER))
        self.uHist = np.zeros((self.dynamics.sysInputDimn, self.TOTAL_ITER))
        self.tHist = np.zeros((1, self.TOTAL_ITER))

        self.xHist1 = np.zeros((self.dynamics1.sysStateDimn, self.TOTAL_ITER))
        self.uHist1 = np.zeros((self.dynamics1.sysInputDimn, self.TOTAL_ITER))
        self.tHist1 = np.zeros((1, self.TOTAL_ITER))

        self.trajectory_file = open("no_liveness_real_time_trajectories_intersection.txt", "w")
        #self.trajectory_file.write("Time, Robot1_x, Robot1_y, Robot1_z, Robot2_x, Robot2_y, Robot2_z\n")
    
    def reset(self):
        """
        Reset the gym environment to its inital state.
        """

        #Reset gym environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds
        self.done = False
        
        #Reset system state
        self.x = self.x0 #retrieves initial condiiton
        self.xObsv = None #reset observer state
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.sysStateDimn, self.TOTAL_ITER))
        self.uHist = np.zeros((self.dynamics.sysInputDimn, self.TOTAL_ITER))
        self.tHist = np.zeros((1, self.TOTAL_ITER))

        self.pointcloudHistory = []
    
    def reset1(self):
        """
        Reset second agent to its inital state.
        """

        #Reset gym environment parameters
        self.iter1 = 0 #number of iterations
        self.t = 0 #time in seconds
        self.done = False
        
        #Reset system state
        self.x1 = self.x01 #retrieves initial condiiton
        self.xObsv1 = None #reset observer state
        
        #Define history arrays
        self.xHist1 = np.zeros((self.dynamics1.sysStateDimn, self.TOTAL_ITER))
        self.uHist1 = np.zeros((self.dynamics1.sysInputDimn, self.TOTAL_ITER))
        self.tHist1 = np.zeros((1, self.TOTAL_ITER))

        self.pointcloudHistory1 = []


    def step(self):
        """
        Step the sim environment by one integration
        """

        position1 = self.dynamics.get_state()[0:3]
        velocity1 = self.dynamics.get_state()[15:18]
        position2 = self.dynamics1.get_state()[0:3]
        velocity2 = self.dynamics1.get_state()[15:18]

        robot2_distance_to_destination = np.linalg.norm([0.5,3,0] - position2.flatten())

        robot1_distance_to_destination = np.linalg.norm([0.5,3,0] - position1.flatten())

        if self.config['remove_after_arriving'] and robot1_distance_to_destination <= self.config['arrive_threshold']:
                if not self.setsingle:
                    qObs = np.array(self.config['orig_qObs']).T
                    rObs = self.config['orig_rObs']

                    vel0 = np.zeros((3, 1))
                    omega0 = np.zeros((3, 1))
                    R0 = np.eye(3).reshape((9, 1))
                    Ts = (self.T*2 - self.t) / 4
                    x01 = self.x1
                    xD1 = np.vstack((np.array([self.config['xD1']]).T, R0, omega0, vel0))

                    self.obstacleManager1 = ObstacleManager(qObs, rObs, NumObs = self.config['numObs'])
                    self.depthManager1 = DepthCamManager(self.observer1, self.obstacleManager1, mean = None, sd = None)
                    self.trajManager1 = TrajectoryManager(x01, xD1, Ts, N = 1)
                    self.controller1 = ControllerManager(self.observer1, QRotorGeometricPD, None, self.trajManager1, self.depthManager1)
                    self.setsingle = True

                self.liveness_value = -1
                
                self._get_observation1()
                self.controller1.set_input(self.t)
                self._update_data1()
                control_input = self.controller1.get_input()

                for i in range(self.SIMS_PER_STEP):
                    self.dynamics1.integrate(control_input, self.t, 1 / self.SIM_FREQ)
                    if self.config['restrict_z']:
                        self.dynamics1._x[2] = [0]
                    self.t += 1 / self.SIM_FREQ 

                robot2_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics1.get_state()[0:3].flatten())
        else:
            if self.iter != 0:
                self._update_managers()

            self._get_observation() #updates the observer
            self.controller.set_input(self.t)
            self._update_data()
            for i in range(self.SIMS_PER_STEP):
                self.dynamics.integrate(self.controller.get_input(), self.t, 1/self.SIM_FREQ) #integrate dynamics
                if self.config['restrict_z']:
                    self.dynamics._x[2] = [0]
                
                self.t += 1/self.SIM_FREQ #increment the time

            robot1_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics.get_state()[0:3].flatten())

            self._update_managers1()
            self._get_observation1() #updates the observer 
            self.controller1.set_input(self.t)
            self._update_data1()
            
            for i in range(self.SIMS_PER_STEP):
                self.dynamics1.integrate(self.controller1.get_input(), self.t, 1/self.SIM_FREQ)
                if self.config['restrict_z']:
                    self.dynamics1._x[2] = [0]
                self.t += 1/self.SIM_FREQ #increment the time 

            robot2_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics1.get_state()[0:3].flatten())
            distance_between_robots = np.linalg.norm(self.x[0:3] - self.x1[0:3])

        self.save_trajectory_step()

    def _update_data(self):
        """
        Update history arrays and deterministic state data
        """
        
        #append the input, time, and state to their history queues
        self.xHist[:, self.iter] = self.x.reshape((self.dynamics.sysStateDimn, ))
        self.uHist[:, self.iter] = (self.controller.get_input()).reshape((self.dynamics.sysInputDimn, ))
        self.tHist[:, self.iter] = self.t
        
        #For Point Cloud from Depth Cam
        # Update point cloud for agent 1 and store it
        ptcloudDict = self.controller.depthManager.get_depth_cam_i(0).get_pointcloud()
        self.pointcloudHistory.append(ptcloudDict["ptcloud"])
        self.pointcloudworldHistory.append(ptcloudDict["ptcloudworld"])

        #For Point Cloud from PointCloud Object
        #update the actual state of the system
        self.x = self.dynamics.get_state()
        
        #update the number of iterations of the step function
        self.iter +=1

    def _update_data1(self):
        """
        Update history arrays and deterministic state data
        """

        #append the input, time, and state to their history queues
        self.xHist1[:, self.iter1] = self.x1.reshape((self.dynamics1.sysStateDimn, ))
        self.uHist1[:, self.iter1] = (self.controller1.get_input()).reshape((self.dynamics1.sysInputDimn, ))
        self.tHist1[:, self.iter1] = self.t
        
        #For Point Cloud from Depth Cam
        # Update point cloud for agent 2 and store it
        ptcloudDict1 = self.controller1.depthManager.get_depth_cam_i(0).get_pointcloud()
        self.pointcloudHistory1.append(ptcloudDict1["ptcloud"])
        self.pointcloudworldHistory1.append(ptcloudDict1["ptcloudworld"])


        #update the actual state of the system
        self.x1 = self.dynamics1.get_state()

        #update the number of iterations of the step function
        self.iter1 +=1
    
    def _update_managers(self):
        """
        Updates obstacleManager, depthManager, and controllerManager
        Necessary for multi agent, as we are treating the other robot as an additional obstacle
        """

        otherRobotpos = (self.x1[0:3].T)[0]
        qObs = np.vstack((self.config['orig_qObs'], otherRobotpos)).T
        rObs = self.config['orig_rObs'] + [self.config['rObs1']]

        vel0 = np.zeros((3, 1))
        omega0 = np.zeros((3, 1))
        R0 = np.eye(3).reshape((9, 1))
        Ts = (self.T*2 - self.t) / 4
        x0 = self.x
        xD = np.vstack((np.array([self.config['xD']]).T, R0, omega0, vel0))

        self.obstacleManager = ObstacleManager(qObs, rObs, NumObs = self.config['numObs'] + 1)
        self.depthManager = DepthCamManager(self.observer, self.obstacleManager, mean = None, sd = None)
        self.trajManager = TrajectoryManager(x0, xD, Ts, N = 1)
        self.controller = ControllerManager(self.observer, QRotorGeometricPD, None, self.trajManager, self.depthManager)


    def _update_managers1(self):
        """
        Updates obstacleManager, depthManager, and controllerManager
        Necessary for multi agent, as we are treating the other robot as an additional obstacle
        """

        otherRobotpos1 = (self.x[0:3].T)[0]
        qObs = np.vstack((self.config['orig_qObs'], otherRobotpos1)).T
        rObs = self.config['orig_rObs'] + [self.config['rObs']]

        Ts = (self.T*2 - self.t) / 4
        x1 = self.x1
        vel1 = np.zeros((3, 1))
        omega1 = np.zeros((3, 1))
        R1 = np.eye(3).reshape((9, 1))
        xD1 = np.vstack((np.array([self.config['xD1']]).T, R1, omega1, vel1))

        self.obstacleManager1 = ObstacleManager(qObs, rObs, NumObs = self.config['numObs'] + 1)
        self.depthManager1 = DepthCamManager(self.observer1, self.obstacleManager1, mean = None, sd = None)
        self.trajManager1 = TrajectoryManager(x1, xD1, Ts, N = 1)
        self.controller1 = ControllerManager(self.observer1, QRotorGeometricPD, None, self.trajManager1, self.depthManager1)

    def _get_observation(self):
        """
        Updates self.xObsv using the observer data
        Useful for debugging state information.

        Original
        
        """

        self.xObsv = self.observer.get_state()

    def _get_observation1(self):
        """
        Updates self.xObsv1 using the observer data
        Useful for debugging state information.

        """

        self.xObsv1 = self.observer1.get_state()
    
    def _get_reward(self):
        """
        Calculate the total reward for ths system and update the reward parameter.
        Only implement for use in reinforcement learning.
        """

        return 0
    
    def _is_done(self):
        """
        Check if the simulation is complete
        Returns:
            boolean: whether or not the time has exceeded the total simulation time
        """

        #check if we have exceeded the total number of iterations
        if self.iter >= self.TOTAL_ITER:
            return True
        return False
    

    def run(self, N=1, run_vis=True, verbose=False, obsManager=None):
        """
        Function to run the simulation N times
        """

        for i in range(N):
            self.reset()
            self.reset1()
            print("Running Simulation.")

            try:
                while not self._is_done():
                    self.step_number += 1
                    if verbose:
                        print("Simulation Time Remaining: ", self.TOTAL_SIM_TIME - self.t)
                    self.step()
            except SimulationComplete:
                print("Simulation terminated early due to a robot reaching its destination.")
                break

        return self.xHist, self.uHist, self.tHist, self.xHist1, self.uHist1, self.tHist1

            
    def visualize(self):
        """
        Provide visualization of the environment
        Inputs:
            obsManager (Obstacle Manager, optional): manager to plot obstacles in the animation
        """

        print("Show interactive Plotly 3D animation with obstacles and robot positions.")
        self.dynamics.show_interactive_3d_animation_plotly(self.xHist, self.xHist1,
                                                            obsManager = self.originalObstacleManager)
    
        # print("Show interactive k3D 3D plot with obstacles and robot positions.")
        # self.dynamics.show_interactive_3d_animation_k3d(self.xHist, self.xHist1,
        #                                                     obsManager = self.originalObstacleManager)
        
        
        # print("Show interactive full k3D 3D animation with obstacles and robot positions.")
        # self.dynamics.show_interactive_k3d_full(self.xHist, self.xHist1,
        #                                                     obsManager = self.originalObstacleManager)
        
        # print("Show interactive full k3D 3D animation using screenshot with obstacles and robot positions.")
        # self.dynamics.show_interactive_k3d_full_with_screenshot(self.xHist, self.xHist1,
        #                                                     obsManager = self.originalObstacleManager)

        # print("Show full animation (Agent pointcloud and robot position):")
        # self.dynamics.show_full_animation(self.xHist, self.xHist1,
        #                                   self.pointcloudHistory, self.pointcloudHistory1,
        #                                   obsManager=self.originalObstacleManager)
        
        # print("Show full animation (World pointcloud and robot position):")
        # self.dynamics.show_full_animation(self.xHist, self.xHist1,
        #                                  self.pointcloudworldHistory, self.pointcloudworldHistory1,
        #                                  obsManager=self.originalObstacleManager)
        
        # print("Show animation basic:")
        # self.dynamics.show_animation_basic(self.xHist,self.xHist1, obsManager = self.originalObstacleManager)
       
        # print("Show animation in visualize (in environment):")
        # self.dynamics.show_animation(self.xHist, self.uHist, self.tHist,
        #                              self.xHist1, self.uHist1, self.tHist1,
        #                              obsManager = self.originalObstacleManager)

        # print("Show plots in visualize:")
        # self.dynamics.show_plots(self.xHist, self.uHist, self.tHist, 
        #                          self.xHist1, self.uHist1, self.tHist1, obsManager = self.originalObstacleManager)

    def save_trajectories_to_file(self, filename="trajectories.txt"):
        """
        Saves the full trajectories of the two agents (robots) to a specified file.
        
        Args:
            filename (str): The name of the file where trajectories will be saved.
        """

        # Calculate the trajectories
        agent1_trajectory = np.vstack((self.xHist[0, :], self.xHist[1, :], self.xHist[2, :])).T
        agent2_trajectory = np.vstack((self.xHist1[0, :], self.xHist1[1, :], self.xHist1[2, :])).T

        # Open the file and write the trajectories
        with open(filename, "w") as file:
            file.write("Agent 1 Trajectory (x, y, z):\n")
            for point in agent1_trajectory:
                file.write(f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}\n")
            
            file.write("\nAgent 2 Trajectory (x, y, z):\n")
            for point in agent2_trajectory:
                file.write(f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}\n")
        
        print("Trajectories saved successfully.")
    
    def save_trajectory_step(self):
        """
        Writes the current position (x, y, z) of both robots, their velocities, relative position and velocity vectors,
        the liveness value, as well as the closest point qC and active CBF region (points with h < 0) to a file.
        """

        # Ensure the file is open for writing (open it in the Environment init)
        if not hasattr(self, 'trajectory_file'):
            self.trajectory_file = open("real_time_trajectories_noliveness.txt", "a")

        if self.config['velocity_metrics']:
            self.velocity_file1 = open("noliveness_robot1_velocity_values.txt", "a")
            self.velocity_file2 = open("noliveness_robot2_velocity_values.txt", "a")

        self.trajectory_file.write(f"Step {self.iter}:\n")

        # Extract current positions and velocities
        agent1_position = self.observer.get_state()[0:3].reshape((3, 1))
        agent1_velocity = self.observer.get_state()[15:18].reshape((3, 1))
        agent2_position = self.observer1.get_state()[0:3].reshape((3, 1))
        agent2_velocity = self.observer1.get_state()[15:18].reshape((3, 1))

        agent1_velocity_mag = np.linalg.norm(agent1_velocity)
        agent2_velocity_mag = np.linalg.norm(agent2_velocity)

        # Calculate relative position and velocity vectors
        relative_position = (agent2_position - agent1_position).flatten()
        relative_velocity = (agent2_velocity - agent1_velocity).flatten()

        #Calculate Robot Distance
        distance_between_robots = np.linalg.norm(relative_position)

        # Set Depth Proc Objects
        depth_proc = self.controller.controllerDict[0].trackingController.depthProc
        depth_proc1 = self.controller1.controllerDict[0].trackingController.depthProc

        # Calculate qC for each agent
        qC = depth_proc.get_closest_point(agent1_position)
        qC1 = depth_proc1.get_closest_point(agent2_position)

        # Retrieve active region points for Agent 1 using DepthProc's eval_cbf_mesh
        mesh_points1 = depth_proc.mesh0 + agent1_position
        h_values1 = depth_proc.eval_cbf_mesh(mesh_points1, self.controller.controllerDict[0].trackingController.h)
        active_points_agent1 = mesh_points1[:, h_values1.flatten() < self.config['active_h_threshold']].T
        active_h_values_agent1 = h_values1[h_values1.flatten() < self.config['active_h_threshold']].flatten()

        # Retrieve active region points for Agent 2 using DepthProc's eval_cbf_mesh
        mesh_points2 = depth_proc1.mesh0 + agent2_position
        h_values2 = depth_proc1.eval_cbf_mesh(mesh_points2, self.controller1.controllerDict[0].trackingController.h)
        active_points_agent2 = mesh_points2[:, h_values2.flatten() < self.config['active_h_threshold']].T
        active_h_values_agent2 = h_values2[h_values2.flatten() < self.config['active_h_threshold']].flatten()

        # Format and write the data to the file
        self.trajectory_file.write(f"Agent 1 Position: {np.round(agent1_position.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 2 Position: {np.round(agent2_position.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 1 Velocity: {np.round(agent1_velocity.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 2 Velocity: {np.round(agent2_velocity.flatten(), 3)}\n")
        if self.config['velocity_metrics']:
            self.velocity_file1.write(f"{agent1_velocity_mag:.3f}\n")
            self.velocity_file2.write(f"{agent2_velocity_mag:.3f}\n")
        self.trajectory_file.write(f"Distance Between Robots: {np.round(distance_between_robots, 3)}\n")
        self.trajectory_file.write(f"Relative Position Vector: {np.round(relative_position, 3)}\n")
        self.trajectory_file.write(f"Relative Velocity Vector: {np.round(relative_velocity, 3)}\n")
        self.trajectory_file.write(f"qC for Agent 1: {np.round(qC[0].flatten(), 3)}\n")
        self.trajectory_file.write(f"qC for Agent 2: {np.round(qC1[0].flatten(), 3)}\n")

        # Write Active Region for Agent 1
        self.trajectory_file.write("Active Region for Agent 1:\n")
        for point, h_val in zip(active_points_agent1, active_h_values_agent1):
            point = np.array(point).reshape((3, 1))
            closest_point = depth_proc.get_closest_point(point)
            self.trajectory_file.write(f"  Point: {np.round(point.flatten(), 3).tolist()}, h: {round(h_val, 3)}, Closest Point: {np.round(closest_point[0].flatten(), 3).tolist()}\n")

        # Write Active Region for Agent 2
        self.trajectory_file.write("Active Region for Agent 2:\n")
        for point, h_val in zip(active_points_agent2, active_h_values_agent2):
            point = np.array(point).reshape((3, 1))
            closest_point = depth_proc1.get_closest_point(point)
            self.trajectory_file.write(f"  Point: {np.round(point.flatten(), 3).tolist()}, h: {round(h_val, 3)}, Closest Point: {np.round(closest_point[0].flatten(), 3).tolist()}\n")

        # Flush to ensure data is written immediately
        self.trajectory_file.flush()

class SimulationComplete(Exception):
    """Custom exception to signal that the simulation should terminate."""
    pass

class EnvironmentWithLiveness(Environment):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.liveness_history = []
        self.liveness_value = 0
        self.setsingle = False
        self.deadlock_threshold = 0.3  # Threshold from the paper
        self.deadlock_steps = 1      # Steps to confirm deadlock
        self.velocity_scaling_factor = 2  # ζ for symmetry breaking
        self.deadlock_count = 0       # Counter for detecting persistent deadlocks
        self.epsilon = 1e-8
        self.zeta = 0.33
        self.trajectory_file = open("real_time_trajectories_intersection.txt", "w")


    def eval_liveness(self, position1, velocity1, position2, velocity2, epsilon=1e-8):
        """
        Evaluate liveness function between two robots based on relative position and velocity.
        Args:
            position1, position2 (numpy array): Positions of the two robots.
            velocity1, velocity2 (numpy array): Velocities of the two robots.
        Returns:
            float: Liveness value in radians.
        """

        relative_position = (position2 - position1).flatten()  # Flatten to 1D
        relative_velocity = (velocity2 - velocity1).flatten()  # Flatten to 1D
        norm_position = np.linalg.norm(relative_position)
        norm_velocity = np.linalg.norm(relative_velocity)
        
        # if norm_position == 0 or norm_velocity == 0:
        #     return np.pi / 2  # Default value for undefined cases
        
        cosine_theta = np.dot(relative_position, relative_velocity) / ((norm_position * norm_velocity) + epsilon)
        cosine_theta = np.clip(cosine_theta, -1.0, 1.0)  # Ensure numerical stability
        theta = np.arccos(cosine_theta)

        if theta > (np.pi)/2:
            return np.pi - theta
        else:
            return theta

    def detect_deadlock(self, liveness_value):
        """
        Detect if the current situation qualifies as a deadlock.
        Args:
            liveness_value (float): Current liveness value.
        Returns:
            bool: True if deadlock is detected, False otherwise.
        """

        if liveness_value <= self.deadlock_threshold:
            self.deadlock_count += 1
            if self.deadlock_count >= self.deadlock_steps:
                return True
        else:
            self.deadlock_count = 0
        return False
    
    

    def calculate_perturbation(self, agent_index):
        """
        Calculates the velocity perturbation for a single robot based on liveliness conditions.
        
        Args:
            robot_state (numpy array): The state of the current robot [x, y, z, ... , v].
            opp_robot_state (numpy array): The state of the opposing robot [x, y, z, ... , v].
            current_input (numpy array): Current control input for the robot (e.g., velocity).
            agent_idx (int): Index of the current robot (0 or 1).
            liveness_threshold (float): Threshold for liveliness condition.
            zeta (float): Scaling factor for desired velocity vector.
        
        Returns:
            numpy array: Adjusted control input after applying the velocity perturbation.
        """

        if agent_index == 0:
            ego_velocity = self.dynamics.get_state()[15:18]
            opp_velocity = self.dynamics1.get_state()[15:18]
        else:
            ego_velocity = self.dynamics1.get_state()[15:18]
            opp_velocity = self.dynamics.get_state()[15:18]

        curr_v0_v1_point = np.array([0.0, 0.0])
        desired_v0_v1_vec = np.array([self.zeta, 1.0])
        desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)

        curr_v0_v1_point = np.array([0.0, 0.0])
        curr_v0_v1_point[agent_index] = np.linalg.norm(ego_velocity)  # Ego velocity
        curr_v0_v1_point[1 - agent_index] = np.linalg.norm(opp_velocity)  # Opponent velocity

        # Desired velocity direction vector
        desired_v0_v1_vec = np.array([self.zeta, 1.0])
        desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)

        # Project the current velocity onto the desired velocity vector
        desired_v0_v1_point = (
            np.dot(curr_v0_v1_point, desired_v0_v1_vec_normalized)
            * desired_v0_v1_vec_normalized
        )

        # Compute the scaling multiplier for the current robot's velocity
        scaling_factor = desired_v0_v1_point[agent_index] / (np.linalg.norm(ego_velocity) + 1e-6)  # Avoid division by zero
        
        return scaling_factor

    def step(self):
        """
        Step the simulation environment by one integration, with deadlock detection and recovery.
        This is the original "alternate" version. It does liveness in the big sim step. 

        it aims to change the control input directly to perturb velocity
        """   
        deadlock_avoidance = False

        # Liveness and deadlock detection between Robot 1 and Robot 2
        position1 = self.dynamics.get_state()[0:3]
        velocity1 = self.dynamics.get_state()[15:18]
        position2 = self.dynamics1.get_state()[0:3]
        velocity2 = self.dynamics1.get_state()[15:18]

        #robot2_distance_to_destination = np.linalg.norm([0.5,3,0] - position2.flatten())
        robot2_distance_to_destination = np.linalg.norm([1,0,0] - position2.flatten())

        #robot1_distance_to_destination = np.linalg.norm([0.5,3,0] - position1.flatten())
        robot1_distance_to_destination = np.linalg.norm([0,1,0] - position1.flatten())


        if robot1_distance_to_destination <= self.config['arrive_threshold']:
            print("Robot 1 has reached its destination.")
            self.save_trajectory_step()
            raise SimulationComplete()
        if self.config['remove_after_arriving'] and robot2_distance_to_destination <= self.config['arrive_threshold']:
                
                if not self.setsingle:
                    print("Robot 2 has reached its destination")
                    qObs = np.array(self.config['orig_qObs']).T
                    rObs = self.config['orig_rObs']

                    vel0 = np.zeros((3, 1))
                    omega0 = np.zeros((3, 1))
                    R0 = np.eye(3).reshape((9, 1))
                    Ts = (self.T*2 - self.t) / 4
                    x0 = self.x
                    xD = np.vstack((np.array([self.config['xD']]).T, R0, omega0, vel0))

                    self.obstacleManager = ObstacleManager(qObs, rObs, NumObs = self.config['numObs'])
                    self.depthManager = DepthCamManager(self.observer, self.obstacleManager, mean = None, sd = None)
                    self.trajManager = TrajectoryManager(x0, xD, Ts, N = 1)
                    self.controller = ControllerManager(self.observer, QRotorGeometricPD, None, self.trajManager, self.depthManager)
                    self.setsingle = True

                self.liveness_value = -1
                
                self._get_observation()
                self.controller.set_input(self.t)
                self._update_data()
                control_input = self.controller.get_input()

                for i in range(self.SIMS_PER_STEP):
                    self.dynamics.integrate(control_input, self.t, 1 / self.SIM_FREQ)
                    #if self.config['restrict_z']:
                    self.t += 1 / self.SIM_FREQ 

                robot1_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics.get_state()[0:3].flatten())
                
        else: 
            liveness = self.eval_liveness(position1, velocity1, position2, velocity2)
            self.liveness_value = liveness
            if self.iter != 0 and self.detect_deadlock(liveness):
                deadlock_avoidance = True
            # Robot 1 update
            if self.iter != 0:
                self._update_managers()
            self._get_observation()
            self.controller.set_input(self.t)
            self._update_data()
            control_input = self.controller.get_input()

            for i in range(self.SIMS_PER_STEP):
                self.dynamics.integrate(control_input, self.t, 1 / self.SIM_FREQ)
                if deadlock_avoidance:
                    robot1_velocity_indices = slice(15, 18)  # Adjust indices based on your state vector structure
                    self.dynamics._x[robot1_velocity_indices] *= self.calculate_perturbation(0)

                if self.config['restrict_z']:
                    self.dynamics._x[2] = [0]
                self.t += 1 / self.SIM_FREQ  

            robot1_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics.get_state()[0:3].flatten())

            # Robot 2 update
            self._update_managers1()
            self._get_observation1()
            self.controller1.set_input(self.t)
            self._update_data1()
            control_input1 = self.controller1.get_input()

            for i in range(self.SIMS_PER_STEP):
                self.dynamics1.integrate(control_input1, self.t, 1 / self.SIM_FREQ)
                # if deadlock_avoidance:
                #     robot1_velocity_indices = slice(15, 18)  # Adjust indices based on your state vector structure
                #     self.dynamics1._x[robot1_velocity_indices] *= self.calculate_perturbation(1)
                if self.config['restrict_z']:
                    self.dynamics1._x[2] = [0]
                self.t += 1 / self.SIM_FREQ
            robot2_distance_to_destination = np.linalg.norm([0.5,3,0] - self.dynamics1.get_state()[0:3].flatten())

            distance_between_robots = np.linalg.norm(self.dynamics.get_state()[0:3] - self.dynamics1.get_state()[0:3])

        self.save_trajectory_step()

    def save_trajectory_step(self):
        """
        Writes the current position (x, y, z) of both robots, their velocities, relative position and velocity vectors,
        the liveness value, as well as the closest point qC and active CBF region (points with h < 0) to a file.
        """
        # Ensure the file is open for writing (open it in the Environment init)
        if not hasattr(self, 'trajectory_file'):
            self.trajectory_file = open("real_time_trajectories.txt", "a")
        if self.config['velocity_metrics']:
            self.velocity_file1 = open("liveness_robot1_velocity_values.txt", "a")
            self.velocity_file2 = open("liveness_robot2_velocity_values.txt", "a")
        if self.config['get_agent2_positions']:
            self.agent2_file = open("agent2_positions_list_intersection.txt", "a")

        self.trajectory_file.write(f"Step {self.iter}:\n")

        # Extract current positions and velocities
        agent1_position = self.observer.get_state()[0:3].reshape((3, 1))
        agent1_velocity = self.observer.get_state()[15:18].reshape((3, 1))
        agent2_position = self.observer1.get_state()[0:3].reshape((3, 1))
        agent2_velocity = self.observer1.get_state()[15:18].reshape((3, 1))

        print("Agent 1 position is: ", agent1_position)
        print("Agent 2 position is: ", agent2_position)

        agent1_velocity_mag = np.linalg.norm(agent1_velocity)
        agent2_velocity_mag = np.linalg.norm(agent2_velocity)

        # Calculate relative position and velocity vectors
        relative_position = (agent2_position - agent1_position).flatten()
        relative_velocity = (agent2_velocity - agent1_velocity).flatten()

        print("Relative Position is: ", relative_position)
        #Calculate Robot Distance
        distance_between_robots = np.linalg.norm(relative_position)

        # Calculate liveness
        liveness_value = self.liveness_value

        # Set Depth Proc Objects
        depth_proc = self.controller.controllerDict[0].trackingController.depthProc
        depth_proc1 = self.controller1.controllerDict[0].trackingController.depthProc

        # Calculate qC for each agent
        qC = depth_proc.get_closest_point(agent1_position)
        qC1 = depth_proc1.get_closest_point(agent2_position)

        #Calculate Distance to qC
        distance_to_qC = np.linalg.norm(qC[0].flatten() - agent1_position.flatten())
        distance_to_qC1 = np.linalg.norm(qC1[0].flatten() - agent2_position.flatten())

        # Calculate Distance to closest static obstacle point
        distance_to_csp = depth_proc.closest_distance_to_static_obstacle(agent1_position)
        distance_to_csp1 = depth_proc.closest_distance_to_static_obstacle(agent2_position)

        #Set Trajectory Objects

        traj = self.trajManager.trajDict[0]
        traj1 = self.trajManager1.trajDict[0]

        desired_position = traj.pos(0.02)
        desired_position1 = traj1.pos(0.02)


        # Retrieve active region points for Agent 1 using DepthProc's eval_cbf_mesh
        mesh_points1 = depth_proc.mesh0 + agent1_position
        h_values1 = depth_proc.eval_cbf_mesh(mesh_points1, self.controller.controllerDict[0].trackingController.h)
        active_points_agent1 = mesh_points1[:, h_values1.flatten() < self.config['active_h_threshold']].T
        active_h_values_agent1 = h_values1[h_values1.flatten() < self.config['active_h_threshold']].flatten()

        # Retrieve active region points for Agent 2 using DepthProc's eval_cbf_mesh
        mesh_points2 = depth_proc1.mesh0 + agent2_position
        h_values2 = depth_proc1.eval_cbf_mesh(mesh_points2, self.controller1.controllerDict[0].trackingController.h)
        active_points_agent2 = mesh_points2[:, h_values2.flatten() < self.config['active_h_threshold']].T
        active_h_values_agent2 = h_values2[h_values2.flatten() < self.config['active_h_threshold']].flatten()

        # Format and write the data to the file
        self.trajectory_file.write(f"Agent 1 Position: {np.round(agent1_position.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 2 Position: {np.round(agent2_position.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 1 Velocity: {np.round(agent1_velocity.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 2 Velocity: {np.round(agent2_velocity.flatten(), 3)}\n")
        if self.config['get_agent2_positions']:
            self.agent2_file.write(f"{np.round(agent2_position.flatten(), 3)}\n")

        if self.config['velocity_metrics']:
            self.velocity_file1.write(f"{agent1_velocity_mag:.3f}\n")
            self.velocity_file2.write(f"{agent2_velocity_mag:.3f}\n")
        self.trajectory_file.write(f"Distance Between Robots: {np.round(distance_between_robots, 3)}\n")
        self.trajectory_file.write(f"Relative Position Vector: {np.round(relative_position, 3)}\n")
        self.trajectory_file.write(f"Relative Velocity Vector: {np.round(relative_velocity, 3)}\n")
        self.trajectory_file.write(f"Liveness Value: {liveness_value:.3f}\n")
        if self.config['dynamic_delta']:
            self.liveness_file.write(f"{liveness_value:.3f}\n")
        self.trajectory_file.write(f"qC for Agent 1: {np.round(qC[0].flatten(), 3)}\n")
        self.trajectory_file.write(f"qC for Agent 2: {np.round(qC1[0].flatten(), 3)}\n")
        self.trajectory_file.write(f"Distance to qC (Agent 1): {np.round(distance_to_qC, 3)}\n")
        self.trajectory_file.write(f"Distance to qC1 (Agent 2): {np.round(distance_to_qC1, 3)}\n")
        self.trajectory_file.write(f"Distance to Closest Static Point (Agent 1): {np.round(distance_to_csp, 3)}\n")
        self.trajectory_file.write(f"Distance to Closest Static Point (Agent 2): {np.round(distance_to_csp1, 3)}\n")

        self.trajectory_file.write(f"Agent 1 Desired Position: {np.round(desired_position.flatten(), 3)}\n")
        self.trajectory_file.write(f"Agent 2 Desired Position: {np.round(desired_position1.flatten(), 3)}\n")

        # Write Active Region for Agent 1
        self.trajectory_file.write("Active Region for Agent 1:\n")
        for point, h_val in zip(active_points_agent1, active_h_values_agent1):
            point = np.array(point).reshape((3, 1))
            closest_point = depth_proc.get_closest_point(point)
            self.trajectory_file.write(f"  Point: {np.round(point.flatten(), 3).tolist()}, h: {round(h_val, 3)}, Closest Point: {np.round(closest_point[0].flatten(), 3).tolist()}\n")

        # Write Active Region for Agent 2
        self.trajectory_file.write("Active Region for Agent 2:\n")
        for point, h_val in zip(active_points_agent2, active_h_values_agent2):
            point = np.array(point).reshape((3, 1))
            closest_point = depth_proc1.get_closest_point(point)
            self.trajectory_file.write(f"  Point: {np.round(point.flatten(), 3).tolist()}, h: {round(h_val, 3)}, Closest Point: {np.round(closest_point[0].flatten(), 3).tolist()}\n")

        # Flush to ensure data is written immediately
        self.trajectory_file.flush()