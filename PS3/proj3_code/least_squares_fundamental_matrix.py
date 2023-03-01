"""Optimize for F using least-squares."""

import numpy as np
from scipy.optimize import least_squares

from proj3_code import two_view_data
from proj3_code.fundamental_matrix import signed_point_line_errors, skew


def objective_function(p, x_0s, x_1s):
    """Objective with new parameterization."""
    F = np.reshape(p, (3, 3))
    return signed_point_line_errors(x_0s, F, x_1s)


def optimize(p0, x_0s, x_1s):
    """Optimize from p0. Make a least_squares() call with 
    fun=objective_function, x0=p0, method='lm', jac='2-point', and
    args=(x_0s, x_1s) as your input. Read the documentation here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    Note: Use the objective function defined above.

    Arguments:
        p0 {ndarray} -- initial estimate of Fundamental Matrix (9, )
        x_0s {ndarray} -- points in image 1, in shape (N, 3)
        x_1s {ndarray} -- points in image 2, in shape (N, 3)

    Returns:
        optimized_F {ndarray} -- resulting estimation of Fundamental Matrix (9, ndarray)
    """
    assert x_0s.shape[1] == 3
    assert x_1s.shape[1] == 3

    optimized_F = None

    ##############################
    # TODO: Student code goes here

    result = least_squares(objective_function, x0 = p0, method='lm', jac='2-point', args=(x_0s, x_1s))
    optimized_F = result.x

    ##############################
    
    return optimized_F


def solve_F(x_0s, x_1s):
    x_0s, x_1s = two_view_data.preprocess_data(x_0s, x_1s)
    p0 = (skew(1, 0, 0)).flatten()  # stereo
    result = optimize(p0, x_0s, x_1s)
    F = np.reshape(result, (3, 3))
    return F

