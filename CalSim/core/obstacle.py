"""
File for obstacles, non-controlled objects with prescribed geometry
"""

class CircularObstacle:
    """
    Class for a circular obstacle. The obstacle should not be interfaced with directly!
    Rather, the obstacle interfaces with a depth camera/lidar object.
    """
    def __init__(self, q, r):
        """
        Init function for an obstacle.
        Inputs:
            q (Nx1 NumPy Array): [X, Y] or [X, Y, Z] position of the obstacle in space
            r (float): radius of obstacle
        """

        #store input parameters
        self._q = q.reshape((q.size, 1)) #reshape to correct (Nx1)
        self._r = r

        #store dimension of shape
        self.dimn = (q.shape)[0]

    def get_center(self):
        """
        Returns the center of the obstacle
        """

        return self._q
    
    def get_radius(self):
        """
        Returns the radius of the obstacle
        """

        return self._r
    
    def set_center(self, qNew):
        """
        Reset the center position of the obstacle
        Inputs:
            qNew (Nx1 NumPy Array): new center position for the obstacle
        """

        self._q = qNew.reshape((qNew.size, 1))
    
    def set_radius(self, rNew):
        """
        Reset the radius of the obstacle
        Inputs:
            rNew (float): new radius for the obstacle
        """

        self._r = rNew


class ObstacleManager:
    def __init__(self, qMatrix, rList, NumObs = 1):
        """
        Managerial class for a set of N obstacles
        Inputs:
            qMatrix (N x NumObs NumPy Array): Matrix containing positions of each obstacle
            rList (list): Python 
            NumObs (Int): number of obstacles
        """

        self.qMatrix = qMatrix
        self.rList = rList
        self.NumObs = NumObs

        #create a dictionary storing N obstacle instances
        self.obsDict = {}

        #create N obstacle objects with a position from qTotal
        for i in range(self.NumObs):
            #create an obstacle with center and radius from qMatrix and rList
            self.obsDict[i] = CircularObstacle(self.qMatrix[:, i], self.rList[i])

    def get_obstacle_i(self, i):
        """
        Function to retrieve the ith obstacle object
        """

        return self.obsDict[i]
    
    def get_obstacle_center_i(self, i):
        """
        Function to retrieve the center of the ith obstacle object
        """

        return self.qMatrix[:, i]