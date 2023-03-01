import time
from typing import Tuple

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


def objective_func(x: np.ndarray, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns it as a 1-D numpy array

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error.
                The parameters are given in the order ([P_11,P_12,P_13,...,P_31,P_32,P_33]).
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error. 
                To get the 2D points, use kwargs['pts2d']
                To get the 3D points, use kwargs['pts3d']
    Returns:
    -     diff: A 1-D numpy array of shape (2*n, ) representing differences between
                projected 2D points and actual 2D points

    """

    diff = None

    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    ##############################
    # TODO: Student code goes here
    n = points_2d.shape[0]
    x = np.append(x, 1) # adding P_34 to the list
    camera_matrix = np.reshape(x, (3,4))

    projected_points = projection(camera_matrix, points_3d)

    diff = points_2d - projected_points
    diff = np.reshape(diff, (2 * n, ))
    ##############################

    return diff


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in non-homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    projected_points_2d = None

    assert points_3d.shape[1]==3

    ##############################
    # TODO: Student code goes here

    points_homogenous = np.insert(points_3d, 3, 1, axis=1)
    
    projected_points_2d = np.matmul(P, points_homogenous.T)

    projected_points_2d = projected_points_2d[0:2] / projected_points_2d[2]
    
    projected_points_2d = projected_points_2d.T
    ##############################

    return projected_points_2d

################# UPDATE function name to estimate_projection_matrix here or update in notebook #########################
def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squares form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points [X_i, Y_i] in image coordinates
        - pts3d: n x 3 array of known points in 3D, [X_i, Y_i, Z_i]
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    P = None

    start_time = time.time()

    kwargs = {'pts2d': pts2d,
              'pts3d': pts3d}


    ##############################
    # TODO: Student code goes here

    initial_guess_reformatted = initial_guess.flatten()[0:11] # remove the last point because it's unnecessary
    result = least_squares(objective_func, initial_guess_reformatted, method='lm', verbose=2, max_nfev=50000, 
        kwargs= kwargs)

    P = np.reshape(np.append(result.x, 1.0), (3, 4))
    
    ##############################

    print("Time since optimization start", time.time() - start_time)

    return P


def decompose_camera_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    K = None
    R = None

    ##############################
    # TODO: Student code goes here

    reshaped = P[:, 0:3]
    K, R = rq(reshaped)
    ##############################

    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix
    -   K: A numpy array of shape (3, 3) representing the intrinsic camera matrix
    -   R_T: A numpy array of shape (3, 3) representing the transposed rotation matrix    
    Returns:
    -   cc: A numpy array of shape (3, ) representing the camera center
            location in world coordinates
    """

    cc = None

    ##############################
    # TODO: Student code goes here
    KR = np.matmul(K, R_T)
    KR_Inverse = np.linalg.inv(KR) 
    cc = -np.matmul(KR_Inverse, P[:, 3])
    ##############################

    return cc
