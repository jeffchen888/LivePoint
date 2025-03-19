"""
This file contains utilities and classes associated with 
the kinematics of manipulator arms.
"""
import numpy as np
from .exceptions import *
from .transforms import *

class Manipulator:
    """
    Class for a manipulator arm. Implements product of exponentials, inverse kinematics,
    jacobian computations.
    """
    def __init__(self, twistList, gst0):
        """
        Init function for classes.
        Inputs:
            twistList (List of Twist objects): List of twists in order from spatial to tool frame
            gst0 (4x4 NumPy Array): base configuration transformation
        """

        #store the twist list
        self._twist_list = twistList
        #store the number of joints
        self.n = len(self._twist_list)
        #store the base transformation
        self._gst0 = gst0

    def compute_fk(self, theta):
        """
        Compute the forward kinematics map of the manipulator
        using the product of exponentials.
        Inputs:
            theta (list of floats): List of joint angles in order from spatial to tool frame
        Returns:
            gst(theta) (4x4 NumPy array): SE(3) transformation from spatial to tool frame
        """
        #check length of input
        if len(theta) != self.n:
            raise NumJointsError()

        #compute the product of the n exponentials
        gst_theta = calc_poe(self._twist_list, theta)

        #multiply by base transformation and return
        return gst_theta @ self._gst0
    
    def compute_xi_i_prime(self, i, theta):
        """
        compute xi_i' for use in the spatial jacobian
        Inputs:
            i (int): index (starting from 1) of the twist we with to transform
            theta (list of floats): list of angles
        """
        #compute the product of exponentials up to i-1
        g_im1 = calc_poe(self._twist_list[0:i-1], theta[0:i-1])
        Ad_gim1 = calc_adjoint(g_im1)

        #multiply the adjoint by xi_i and return
        xi_i = self._twist_list[i-1].get_xi()
        return Ad_gim1 @ xi_i
    
    def compute_spatial_jacobian(self, theta):
        """
        Compute the spatial jacobian of a manipulator arm at 
        configuration theta = [theta1, theta2, ..., thetan]
        Inputs:
            theta (list of floats): List of joint angles in order from spatial to tool frame
        Returns:
            Jst_s(theta) (6xn NumPy Array): Spatial manipulator jacobian
        """
        #check length of input
        if len(theta) != self.n:
            raise NumJointsError()

        #initialize Jst_s with xi_1
        Jst_s = self._twist_list[0].get_xi()

        #loop over the remaining twists
        for i in range(1, self.n):
            #compute xi_i' - remember to shift to 1-indexing
            xi_i_prime = self.compute_xi_i_prime(i+1, theta)
            #update the spatial jacobian
            Jst_s = np.hstack((Jst_s, xi_i_prime))

        #return the spatial jacobian
        return Jst_s
    
    def compute_body_jacobian(self, theta):
        """
        Compute the body jacobian of a manipulator arm at 
        configuration theta = [theta1, theta2, ..., thetan]
        Inputs:
            theta (list of floats): List of joint angles in order from spatial to tool frame
        Returns:
            Jst_b(theta) (6xn NumPy Array): Body manipulator jacobian
        """
        #calculate the transformation gst(theta) and its adjoint
        gst_theta = self.compute_fk(theta)
        ad_gst = calc_adjoint(gst_theta)

        #calculate the spatial jacobian and extract the body jacobian from the adjoint
        Jst_b = np.linalg.inv(ad_gst)@self.compute_spatial_jacobian(theta)
        return Jst_b