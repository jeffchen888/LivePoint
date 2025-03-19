"""
This file implements the twist class.
"""
import numpy as np
from .transforms import *

class Twist:
    def __init__(self, v, omega):
        """
        Init function for a unit twist
        Inputs:
            v (3x1 NumPy Array): linear component of twist
            omega (3x1 NumPy Array): angular component of twist
        """
        self._v = v.reshape((3, 1))
        self._omega = omega.reshape((3, 1))
        self._xi = np.vstack((self._v, self._omega))

    def get_v(self):
        """
        Returns linear component of twist
        """
        return self._v
    
    def get_omega(self):
        """
        Returns angular component of twist
        """
        return self._omega
    
    def get_xi(self):
        """
        Returns 6x1 (v, omega)
        """
        return self._xi
    
    def set_v(self, v):
        #reset v
        self._v = v.reshape((3, 1))
        #reset xi
        self._xi = np.vstack((self._v, self._omega))
    
    def set_omega(self, omega):
        #reset omega
        self._omega = omega.reshape((3, 1))
        #reset xi
        self._xi = np.vstack((self._v, self._omega))

    def set_xi(self, v, omega):
        self._v = v.reshape((3, 1))
        self._omega = omega.reshape((3, 1))
        self._xi = np.vstack((self._v, self._omega))

    def exp(self, theta):
        """
        Calculate the matrix exponential of the twist with 
        magnitude theta.
        Inputs:
            theta (float): magnitude (linear distance/angle in radians)
        Returns:
            g ((4x4) NumPy Array): SEE(3) transformation
        """
        return exp_transform(self._xi, theta)