"""
Hybrid Domain Class -> Defines object corresponding to a single hybrid domain
"""

class HybridDomain:
    """
    Hybrid domain object. Represents each Di of a Hybrid system.
    Contains associated dynamics, guard, and transition functions
    """
    def __init__(self, f, SList, DeltaList, domainIndex):
        """
        Init function for a hybrid domain Di
        Inputs:
            f (function): continuous dynamics associated with the domain
            SList (List of functions): List of guard functions associated with the domain
            DeltaList (List of functions): List of reset functions associated with the domain (same order as SList)
            domainIndex (int): Integer of the INDEX of the domain Di
        """   
        self.f = f
        self.SList = SList #each element should be a function of the form s(x, u, t) -> returns True or False based on guard condition
        self.DeltaList = DeltaList #each element should be a function of the form Delta(x, u, t) -> returns new state AND index of new domain
        self.i = domainIndex

    def set_dyn(self, f):
        """
        Set the continuous dynamics xDot = f(x, u, t) associated with the domain
        """
        self.f = f
    
    def set_SList(self, SList):
        """
        Setter function for S list, list of guard functions.
        Each guard function should have inputs (x, u, t) and should
        return True if guard condition met and False otherwise.
        -> s(x, u, t) = True if guard met, s(x, u, t) False if guard not
        """
        self.SList = SList

    def set_DeltaList(self, DeltaList):
        """
        Setter function for Delta list, list of transition functions corresponding to the guard list.
        Each transition function should have inputs (x, u, t) and should
        return the next state x+ after transition and the DOMAIN INDEX of the state we transition to. 
        Note: 
        -> s(x, u, t) = True if guard met, s(x, u, t) False if guard not
        """
        self.DeltaList = DeltaList

    def set_index(self, i):
        """
        Sets the domain index (i.e. the identifying number) of the domain
        """
        self.i = i

    def get_dyn(self):
        return self.f
    
    def get_SList(self):
        return self.SList
    
    def get_DeltaList(self):
        return self.DeltaList
    
    def get_index(self):
        return self.i