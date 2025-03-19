"""
This file includes various classes and functions involved in rigid body transformations.
"""
import numpy as np
from .exceptions import *

def hat_3d(w):
    """
    Function to compute the hat map of a 3x1 vector omega.
    Inputs:
        w (3x1 NumPy Array): vector to compute the hat map of
    Returns:
        w_hat (3x3 NumPy Array): skew symmetrix hat map matrix
    """
    #reshape the w vector to be a 3x1
    w = w.reshape((3, 1))
    #compute and return its hat map
    w_hat = np.array([[0, -w[2, 0], w[1, 0]], 
                      [w[2, 0], 0, -w[0, 0]], 
                      [-w[1, 0], w[0, 0], 0]])
    return w_hat

def vee_3d(wHat):
    """
    Function to compute the vee map of a 3x3 matrix in so(3).
    Inputs:
        wHat (3x3 NumPy Array): matrix in so(3) to compute vee map of
    Returns:
        w (3x1 NumPy Array): 3x1 vector corresponding to wHat
    """
    return np.array([[wHat[2, 1], wHat[0, 2], wHat[1, 0]]]).T

def hat_6d(xi):
    """
    Function to compute the hat map of a 6x1 twist xi
    Inputs:
        xi (6x1 NumPy Array): (v, omega) twist
    Returns:
        xi_hat (4x4 NumPy Array): hat map matrix
    """
    #reshpae to a 6x1 Vector
    xi = xi.reshape((6, 1))
    #compute the hat map of omega
    w_hat = hat_3d(xi[3:])
    #extract and reshape v
    v = xi[0:3].reshape((3, 1))
    #compute and return the hat map of xi
    xi_hat = np.hstack((w_hat, v))
    xi_hat = np.vstack((xi_hat, np.zeros((1, 4))))
    return xi_hat

def hat(x):
    """
    Function to compute the hat map of a 6x1 or 3x1 vector x
    Inputs:
        x (6x1 or 3x1 NumPy Array)
    Returns:
        x_hat: hat map of the vector x
    """
    if x.size == 3:
        return hat_3d(x)
    elif x.size == 6:
        return hat_6d(x)
    else:
        #raise error: input vector is of incorrect shape
        raise ShapeError()

def rodrigues(w, theta):
    """
    Function to compute the matrix exponential of an angular velocity vector
    using rodrigues' formula.
    Inputs:
        w (3x1 NumPy Array): angular velocity vector (may be unit or not unit)
        theta (float): angle of rotation in radians
    Returns:
        exp(w_hat*theta): rotation matrix associated with w and theta
    """
    #check shape of w
    if w.size != 3:
        raise ShapeError()
    
    #reshape w
    w = w.reshape((3, 1))

    #compute Rodrigues formula (using the non-unit w assumption)
    wNorm = np.linalg.norm(w)
    wHat = hat(w)
    exp_w_theta = np.eye(3) + (wHat/wNorm)*np.sin(wNorm*theta) + (wHat@wHat)/(wNorm**2)*(1-np.cos(wNorm*theta))
    
    #return matrix exponential
    return exp_w_theta

def calc_Rx(phi):
    """
    Function to copute the X Euler angle rotation matrix
    Inputs:
        phi (float): angle of rotation
    Returns:
        R_x(phi) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[1, 0, 0]]), phi)

def calc_Ry(beta):
    """
    Function to copute the Y Euler angle rotation matrix
    Inputs:
        beta (float): angle of rotation
    Returns:
        R_y(beta) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[0, 1, 0]]), beta)

def calc_Rz(alpha):
    """
    Function to copute the Z Euler angle rotation matrix
    Inputs:
        alpha (float): angle of rotation
    Returns:
        R_z(alpha) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[0, 0, 1]]), alpha)

def calc_Rzyz(alpha, beta, gamma):
    """
    Calculate a rotation matrix based on ZYZ Euler angles
    Inputs:
        alpha, beta, gamma: rotation angles
    Returns:
        Rz(alpha)Ry(beta)Rz(gamma)
    """
    return calc_Rz(alpha)@calc_Ry(beta)@calc_Rz(gamma)

def calc_Rzyx(psi, theta, phi):
    """
    Calculate a rotation matrix based on ZYX Euler angles
    Inputs:
        psi, theta, phi: rotation angles
    Returns:
        Rz(psi)Ry(theta)Rx(phi)
    """
    return calc_Rz(psi)@calc_Ry(theta)@calc_Rx(phi)


def quat_2_rot(Q):
    """
    Calculate the rotation matrix associated with a unit quaternion Q
    Inputs:
        Q (4x1 NumPy Array): [q, q0] unit quaternion, where q is 3x1 and q0 is scalar
    Returns:
        R (3x3 NumPy Array): rotation matrix associated with quaternion
    """
    #extract the quaternion components
    b, c, d, a = Q.reshape((4, ))
    #compute and return the rotation matrix
    R = np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c], 
                  [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b], 
                  [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]])
    return R

def exp_twist(xi, theta):
    """
    Calculate the matrix exponential of a unit twist, xi
    Inputs:
        xi (6x1 NumPy Array): unit twist
        theta (float): magnitude of transformation
    Returns:
        exp(xi_hat*theta) (4x4 NumPy Array): SE(3) transformation
    """
    #reshape xi
    xi = xi.reshape((6, 1))

    #extract v and omega
    v = xi[0:3].reshape((3, 1))
    w = xi[3:].reshape((3, 1))

    #compute exponential
    if np.linalg.norm(w) == 0:
        #case of zero omega
        R = np.eye(3)
        p = v*theta
    else:
        #case of nonzero omega
        R = rodrigues(w, theta)
        p = (np.eye(3) - R)@(hat(w) @ v) + w @ w.T @ v * theta
    
    #compute the blocks of the transformation
    expXi = np.hstack((R, p))
    expXi = np.vstack((expXi, np.array([[0, 0, 0, 1]])))
    return expXi

def exp_transform(x, theta):
    """
    Computes an SO(3) or SE(3) transformation of a unit axis/twist x with 
    magnitude theta using the closed forms of the matrix exponential.
    Inputs:
        x (3x1 or 6x1 NumPy Array): axis or twist
        theta (scalar): magnitude of transformation
    Returns:
        g (4x4 NumPy Array) or R (3x3 NumPy Array): SE(3) or SO(3) transformation
    """
    if x.length == 3:
        #return SO(3) transformation
        return rodrigues(x, theta)
    elif x.length == 6:
        #return SE(3) transformation
        return exp_twist(x, theta)
    else:
        raise ShapeError()

def calc_adjoint(g):
    """
    Calculate the adjoint matrix of a transformation g in SE(3)
    Inputs:
        g (4x4 NumPy Array): SE(3) transformation matrix
    Returns:
        Adg (6x6 NumPy Array): Adjoint of the transformation
    """
    #extract the rotation matrix
    R = g[0:3, 0:3]
    p = g[0:3, 3]

    #compute the blocks of the adjoint
    upperBlocks = np.hstack((R, hat(p)@R))
    lowerBlocks = np.hstack((np.zeros((3, 3)), R))
    return np.vstack((upperBlocks, lowerBlocks))

def calc_poe(twist_list, theta_list):
    """
    Compute the product of exponentials:
    exp(xi_1*theta_1)*exp(xi_2*theta_2)...*exp(xi_n*theta_n)
    Inputs:
        twist_list (list of Twist objects)
        theta_list (list of angles)
    Returns:
        g_theta (4x4 NumPy Array): SE(3) transformation exp(xi_1*theta_1)*...*exp(xi_n*theta_n)
    """
    #initialize transformation
    g_theta = np.eye(4)

    #compute the transformation using POE
    for i in range(len(twist_list)):
        xi_i = twist_list[i]
        g_theta = g_theta @ xi_i.exp(theta_list[i])
    
    return g_theta
    