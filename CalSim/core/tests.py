from control import *

#tests for the control functions
def peak_time_test():
    #test peak time
    tData = np.array(range(629))
    yData = np.array([np.sin(x) for x in np.arange(0, 2*np.pi, 0.01)])
    Tp = calc_peak_time(yData, tData)
    assert(np.array_equal(Tp, np.array([[157]])))


peak_time_test()
print("Everything passed!")