"""
    Correlation Coefficients

"""

import numpy as np
from scipy.optimize import leastsq


def calc_cc(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    return np.corrcoef(x, y)


def mleastsq(x, y):
    return leastsq(x, y)

