"""
This file contains utilities for control design.
"""
import numpy as np
from scipy.signal import find_peaks

def ctrb(A, B):
    """
    Function to compute the controllability matrix of a system xDot = Ax + Bu
    Inputs:
        A (nxn NumPy Array): A matrix of a system
        B (nxm NumPy Array): B matrix of a system
    Returns:
        [B AB A^2B ... A^(n-1)B]
    """
    #initialize controllabiity matrix as B
    P = B
    for i in range(1, A.shape[0]):
        P = np.hstack((P, np.linalg.matrix_power(A, i) @ B))
    #return controllability matrix
    return P

def obsv(A, C):
    """
    Function to compute the observability matrix of a system xDot = Ax + Bu, y = Cx
    Inputs:
        A (nxn NumPy Array): A matrix of a system
        C (mxn NumPy Array): C matrix of a system
    Returns:
        [C; CA; CA^2; ...; CA^n-1]
    """
    #initialize observability matrix as C
    O = C
    for i in range(1, A.shape[0]):
        P = np.vstack((P, C @ np.linalg.matrix_power(A, i)))
    #return observability matrix
    return O

def is_ctrb(A, B):
    """
    Verify if (A, B) is a controllable pair. Returns true if controllable.
    """
    return np.linalg.matrix_rank(ctrb(A, B)) == A.shape[0]

def is_obsv(A, C):
    """
    Verify if (A, C) is an observable pair. Returns true if observable.
    """
    return np.linalg.matrix_rank(ctrb(A, C)) == A.shape[0]

def calc_cl_poles(A, B, K):
    """
    Function to calculate the closed loop poles of a system xDot = Ax + Bu
    using state feedback with gain matrix K.
    Inputs:
        A (nxn NumPy Array): A matrix
        B (nxm NumPy Array): B Matrix
        K (mxn NumPy Array): Gain matrix
    Returns:
        [lambda1, lambda2, ..., lambdan] (list of floats): closed loop poles of the system with gain K
    """
    return np.linalg.eigvals(A - B @ K).tolist()

def place_poles(A, B, pole_list):
    """
    Function to compute the gain K to place the poles of A - BK at desired posiions using Ackermann's formula.
    This function depends on invertibility of the controllability matrix.
    Inputs:
        A (nxn NumPy Array): A matrix
        B (nxm NumPy Array): B Matrix
        pole_list (list of n complex/float numbers): list of desired pole positions for the closed loop system
    Returns:
        K (mxn NumPy Array): State feedback gain matrix to place the poles of the system in the desired locations
    """
    #find the char. polyn of A
    char_poly = np.poly(np.linalg.eigvals(A))

    #find the desired char. polyn
    char_poly_des = np.poly(pole_list)

    #subtract the desired poles
    rowVec = (char_poly_des[1:] - char_poly[1:]).reshape((1, A.shape[0]))

    #compute the W terms
    Wr = ctrb(A, B)

    #assemble Wr tilda (inverse of toeplitz matrix)
    Wrt = np.eye(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j - i >= 0:
                #check the relation between i and j
                Wrt[i, j] = char_poly[j - i]
            else:
                Wrt[i, j] = 0
    Wrt = np.linalg.pinv(Wrt)

    #return the gain
    return rowVec @ np.linalg.pinv(Wr) @ Wrt

def calc_first_peak(yData):
    """
    Function to find first peak and their indices of a signal yData.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
    Returns
        yPeaks (m x 1 NumPy Array): vector of first peak values
        yPeakIndices (m x 1 NumPy Array): vector of indices of the first peaks
    """
    #reshape yData to have two dimensions
    if len(yData.shape) == 1:
        #reshape yData to be a row vector
        yData = yData.reshape((1, yData.shape[0]))

    #initialize an empty peak and peak index array
    yPeaks = []
    yPeakIndices = []
    
    for i in range(yData.shape[0]):
        #slice out the row of outputs
        yiData = yData[i, :]

        #find the indices of the local maxima in yData -> take the first local maximum
        peakIndices, _ = find_peaks(yiData)

        #append the value of y at the first peak index
        yPeaks.append(yiData[peakIndices[0]])

        #append the first peak index
        yPeakIndices.append(peakIndices[0])

    #convert to NumPy Arrays and return
    return np.array([yPeaks]).T, np.array([yPeakIndices]).T

def calc_peak_time(yData, tData):
    """
    Function to calculate the peak time of a system's output. Note: this function 
    will generally only converge for step responses for linear systems. It is designed
    around a typical linear second order response for a stable system.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
        tData (1 x N NumPy Array): time series corresponding to yData
    Returns:
        Tp (m x 1 NumPy array): Peak time for each output
    """
    #calculate the indices of the peaks
    _, peakIndices = calc_first_peak(yData)

    #initialize an empty rise time array
    peakTimeArr = []

    #find the times associated with the first peaks
    for i in range(peakIndices.shape[0]):
        #get the time associated with the peak index
        peakTimeArr.append(tData[peakIndices[i, 0]])

    #convert riseTimeArray to a NumPy array and return
    return np.array([peakTimeArr]).T

def calc_ss_value(yData):
    """
    Function to estimate the steady state value of a signal. This will only return a 
    good estimate if the settling time has been reached within tData and the response is stable.
    Returns the vector of final values of the signal.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
    Returns
        ySS (m x 1 NumPy Array): vector of final output values
    """
    #reshape yData to have two dimensions
    if len(yData.shape) == 1:
        #reshape yData to be a row vector
        yData = yData.reshape((1, yData.shape[0]))

    #slice out last column in yData and return
    return yData[:, -1].reshape((yData.shape[0], 1))

def find_percent_index(yData, ySS, percent):
    """
    Function to calculate the index at which the output reaches a 
    particular percent of its SS value. Used as a helper function for rise time.
    Moves from the front of the signal to the back. (Data is potentially unordered so cannot binary search)
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
        ySS (m x 1 NumPy Array): steady state value array
        percent (float): float from 0 to 100 describing desired percentage to find
    Returns:
        indexList (list of m ints): list of indices at which the percentage is achieved
    """
    #reshape yData to have two dimensions
    if len(yData.shape) == 1:
        #reshape yData to be a row vector
        yData = yData.reshape((1, yData.shape[0]))

    #initialize an index list
    indexList = [0]*yData.shape[0]
    for i in range(yData.shape[0]):
        #search from the beginning of the signal to the end for the cutoff
        for j in range(yData.shape[1]):
            #if we pass the perentage threshold, update indexList and break inner loop
            if abs((yData[i, j] - ySS[i, 0])/ySS[i, 0]) > percent/100:
                indexList[i] = j
                break
    #return resulting list of indices
    return indexList

def calc_rise_time(yData, tData):
    """
    Function to calculate the rise time of a system's output, i.e. the time 
    it takes for the response to go from 10% to 90% of its SS value.
    Note: this function will generally only converge for step responses for linear systems. 
    It is designed around a typical linear second order response for a stable system.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
        tData (1 x N NumPy Array): time series corresponding to yData
    Returns:
        Tr (m x 1 NumPy array): Rise time for each output
    """    
    #calculate the SS values for each output
    ySS = calc_ss_value(yData)

    #find the 10% and 90% values we're looking for
    indexList10 = find_percent_index(yData, ySS, 10)
    indexList90 = find_percent_index(yData, ySS, 90)

    #initialize an empty rise time array
    TrArr = []

    #find the times associated with each and subtract to get rise time
    for i in range(len(indexList10)):
        TrArr.append(tData[indexList90[i]] - tData[indexList10[i]])

    #return the rise time array
    return np.array([TrArr]).T

def calc_percent_os(yData):
    """
    Function to calculate the percent overshoot of a system's output. Note: this function 
    will generally only converge for step responses for linear systems. It is designed
    around a typical linear second order response for a stable system.
    For this function to work, it is important that y has settled to its SS response at the end of tData.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
    Returns:
        %OS (m x 1 NumPy array): Percent overshoot (from 0 to 100) for each output
    """    
    #calculate the peak values
    peakArray, _ = calc_first_peak(yData)

    #calculate the steady state values
    ssArray = calc_ss_value(yData)

    #find the percent overshoot for each element and return
    return 100*np.divide(peakArray - ssArray, ssArray)

def settling_time_helper(y, yss):
    """
    Helper function for settling time. Checks if the +-2% condition is broken for an output value y & SS value yss
    Inputs:
        y (float): scalar output value to evaluate
        yss (float): steady state value
    Returns:
        True/False (boolean): if condition has been broken or not for this particular y
    """
    checkVal = abs((y-yss)/yss)
    return (0.98 <= checkVal) and (checkVal <= 1.02)

def calc_settling_time(yData, tData):
    """
    Function to calculate the settling time of a response, i.e. the time to get and stay within +-2% of the SS value.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
        tData (1 x N NumPy Array): time series corresponding to yData
    Returns:
        Ts (m x 1 NumPy array): Settling time for each output
    """
    #reshape yData to have two dimensions
    if len(yData.shape) == 1:
        #reshape yData to be a row vector
        yData = yData.reshape((1, yData.shape[0]))

    #calculate SS values
    ySS = calc_ss_value(yData)

    #initialize an empty settling time array
    TsArr = [0]*ySS.shape[0]

    #search for +-2% index. If none is found, will simply return the final index.
    for i in range(ySS.shape[0]):
        #move backwards through each time series from the end to the beginning
        for j in range(reversed(yData.shape[1])):
            #Add the time to TsArray
            TsArr[i] = tData[j]
            #evaluate if this particular timestep satisfies the +-2% condition
            if not settling_time_helper(yData[i, j], ySS[i]):
                #break the inner loop
                break
    #convert to NumPy array and return
    return np.array([TsArr]).T

def response_info(yData, tData):
    """
    Master function to provide information about the response of the system. Prints and returns
    specifications about the response contained in yData.
    Inputs:
        yData (m x N NumPy Array): time series output of the system (rows are outputs, columns are at different time steps)
        tData (1 x N NumPy Array): time series corresponding to yData
    Returns:
        Tr, Ts, OS, Tp: Rise time, Settling Time, Percent Overshoot, Peak Time NumPy arrays
    """
    #calculate the response specifications
    Tr = calc_rise_time(yData, tData)
    Ts = calc_settling_time(yData, tData)
    Overshoot = calc_percent_os(yData)
    Tp = calc_peak_time(yData, tData)

    #print the response information
    print("Rise Time (s): ", Tr.T)
    print("Settling Time (s):", Ts.T)
    print("Percent Overshoot (%)", Overshoot.T)
    print("Peak Time (s): ", Tp.T)

    #return the response information
    return Tr, Ts, Overshoot, Tp