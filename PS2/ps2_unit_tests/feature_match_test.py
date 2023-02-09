#!/usr/bin/python3

import numpy as np
from sklearn.decomposition import PCA

from ps2_code.student_feature_matching import match_features, compute_feature_distances, pca


def test_compute_dists():
    """
    Test feature distance calculations.
    """
    feats1 = np.array(
        [
            [.707, .707],
            [-.707, .707], 
            [-.707, -.707]
        ])
    feats2 = np.array(
        [
            [-.5, -.866],
            [.866, -.5],
            [.5, .866],
            [-.866, .5]
        ])
    dists = np.array(
        [
            [1.98271985, 1.21742762, 0.26101724, 1.58656169],
            [1.58656169, 1.98271985, 1.21742762, 0.26101724],
            [0.26101724, 1.58656169, 1.98271985, 1.21742762]
        ])
    inter_distances = compute_feature_distances(feats1, feats2)
    assert inter_distances.shape[0] == 3
    assert inter_distances.shape[1] == 4
    assert np.allclose(dists, inter_distances, atol = 1e-03)


def test_feature_matching():
    """
    Few matches example. Match based on the following affinity/distance matrix:

        [2.  1.2 0.3 1.6]
        [1.6 2.  1.2 0.3]
        [0.3 1.6 2.  1.2]
        [1.2 0.3 1.6 2. ]
    """
    feats1 = np.array(
        [
            [.707, .707],
            [-.707, .707],
            [-.707, -.707],
            [.707, -.707]
        ])
    feats2 = np.array(
        [
            [-.5, -.866],
            [.866, -.5],
            [.5, .866],
            [-.866, .5]
        ])
    x1 = np.array([11,12,13,14])
    y1 = np.array([14,13,12,11])
    x2 = np.array([11,12,13,14])
    y2 = np.array([15,16,17,18])
    matches = np.array(
        [
            [0,2],
            [1,3],
            [2,0],
            [3,1]
        ])
    result, confidences = match_features(feats1, feats2)
    assert np.array_equal(matches, result[np.argsort(result[:, 0])])

    

def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs

def test_pca():
    dummy1 = np.array([[34, 85, 79, 30, 84, 16, 16, 46, 90, 58],
       [51, 71, 60, 21, 61, 46,  1, 13, 91, 55],
       [42, 14, 12, 89, 73,  9, 51, 58, 14, 46],
       [81, 63, 59, 17, 59, 13, 88, 74, 99, 29],
       [46, 50,  2, 47, 88, 18, 32, 83, 84, 51],
       [39, 21, 36, 67, 87,  6, 61, 14, 13, 87],
       [66, 51, 69, 12, 87, 41, 31, 84, 25, 85],
       [73, 15, 81, 23, 52, 62, 86, 40, 98, 11],
       [79,  5, 38, 30, 17, 25, 74, 68, 25, 26],
       [50, 65, 21, 55, 51, 64,  1, 29, 45, 90],
       [73, 58, 15,  3, 44,  7,  3, 10,  4, 12],
       [12, 54,  6,  7,  9, 38, 86, 62, 48, 87],
       [99, 75, 55, 89, 25,  9,  8, 69, 47, 23],
       [49, 19, 52, 66, 40, 91, 78, 26, 11, 20],
       [17, 98, 67, 56, 12, 12, 88, 99, 26, 91],
       [64, 17, 70,  4, 46, 48, 44, 28,  1, 96],
       [54, 57, 66, 71, 79, 90, 69,  8, 90, 99],
       [33, 14, 62, 13, 78, 57, 50, 76, 53,  5],
       [53,  4, 90, 20,  8, 39, 47,  3, 37, 37],
       [ 2, 72, 87, 90, 86, 81, 32, 15, 49, 41],
       [ 2, 50,  0, 66, 93, 78, 89, 86, 11, 56],
       [65, 47, 55, 47, 80, 36, 99, 82,  5, 54],
       [66, 17, 96, 67, 47, 12, 82, 93, 35, 13],
       [63, 75, 22, 70, 94,  7,  7, 55, 19, 35],
       [48,  2, 14,  7, 39,  9, 94, 24,  9,  8]])

    dummy2 = np.array([[96, 24, 63,  4,  1, 83, 74, 50, 62, 56],
       [ 5, 32, 35, 91, 54, 69, 99, 49, 56,  1],
       [71, 86, 51, 28,  9, 91, 83, 63,  2, 55],
       [98, 82, 99, 29, 80, 26, 61, 10, 26,  3],
       [74, 93, 86, 21, 79, 52, 11, 10, 69, 33],
       [ 7, 38,  8, 81, 63, 29, 66, 16, 14, 30],
       [34, 39,  2, 51, 10, 53, 42, 20, 19, 84],
       [69, 96, 90,  0, 15, 46, 55, 33, 65, 77],
       [97, 95, 43, 96, 85, 82, 83, 80, 85, 99],
       [15, 59, 68,  4,  1, 57, 23, 71, 72, 93],
       [92, 11, 62, 15, 10, 73, 45, 47, 71, 22],
       [74, 43, 59, 98, 43, 61, 37, 93, 90, 39],
       [26, 45, 57, 23, 42, 64, 66, 95, 33, 93],
       [96, 76, 87,  4, 14, 53, 73, 93, 52, 10],
       [82, 69, 30, 25, 21, 86, 86, 28, 33, 80],
       [39, 17,  9,  1, 75, 41, 98, 29, 32, 20],
       [49,  4, 40, 93, 72,  8, 84, 98, 68,  9],
       [43, 82, 59, 22, 96, 45, 87, 69, 50, 48],
       [31, 24, 21, 17, 23, 31, 88, 47, 18, 77],
       [75, 23, 89, 88, 46, 47, 18, 89, 62, 19],
       [41, 18, 99, 90,  9,  2, 70,  6, 92, 90],
       [ 8, 23, 23, 43, 46, 56, 66, 82, 43, 45],
       [31, 47, 66, 53,  7, 88, 13, 25, 54, 17],
       [64, 39, 75, 49, 36, 95, 21, 87, 50, 63]])

    red_feats1, red_feats2 = pca(dummy1, dummy2, 3)
    pca_true = PCA(3)
    pca_true.fit(np.vstack((dummy1, dummy2)))
    red_feats1_true = pca_true.transform(dummy1)
    red_feats2_true = pca_true.transform(dummy2)

    assert np.allclose(*flip_signs(red_feats1_true, red_feats1), atol=1e-4) and np.allclose(*flip_signs(red_feats2_true, red_feats2), atol=1e-4)




