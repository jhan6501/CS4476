import numpy as np
import sklearn.neighbors

def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array to every feature in another.
    
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of features 
    - features2: A numpy array of shape (m,feat_dim) representing the second set features
      
    Note: n, m is the number of feature (m not necessarily equal to n); 
          feat_dim denotes the feature dimensionality;
    
    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`compute_feature_distances` function in ' +
        '`student_feature_matching.py` needs to be implemented')
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).
    To start with, simply implement the NNDR, "ratio test", which is the equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).
    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Step:
    1. Use `compute_feature_distances()` to find out the distance
    2. Implement the NNDR equation to find out the match
    3. Record the match indecies ('matches') and distance of the match ('Confidences')
    
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of features 
    - features2: A numpy array of shape (m,feat_dim) representing the second set features
      
    Note: n, m is the number of feature (m not necessarily equal to n); 
          feat_dim denotes the feature dimensionality;

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match, which is the distance between matched pair.
    
    E.g. The first feature in 'features1' matches to the third feature in 'features2'.  
         Then the output value for 'matches' should be [0,2] and 'confidences' [0.9]

    Note: 'matches' and 'confidences' can be empty which has shape (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`match_features` function in ' +
        '`student_feature_matching.py` needs to be implemented')
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences

def pca(fvs1, fvs2, n_components= 24):
    """
    Perform PCA to reduce the number of dimensions in each feature vector which resulting in a speed up.
    You will want to perform PCA on all the data together to obtain the same principle components.
    You will then resplit the data back into image1 and image2 features.

    Helpful functions: np.linalg.svd, np.mean, np.cov

    Args:
    -   fvs1: numpy nd-array of feature vectors with shape (n,128) for number of interest points 
        and feature vector dimension of image1
    -   fvs1: numpy nd-array of feature vectors with shape (m,128) for number of interest points 
        and feature vector dimension of image2
    -   n_components: m desired dimension of feature vector

    Return:
    -   reduced_fvs1: numpy nd-array of feature vectors with shape (n, m) with m being the desired dimension for image1
    -   reduced_fvs2: numpy nd-array of feature vectors with shape (n, m) with m being the desired dimension for image2
    """

    reduced_fvs1, reduced_fvs2 = None, None
    #############################################################################
    # TODO: YOUR PCA CODE HERE                                                  #
    #############################################################################

    raise NotImplementedError('`pca` function in ' +
    '`student_feature_matching.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return reduced_fvs1, reduced_fvs2

def accelerated_matching(features1, features2):
    """
    This method should operate in the same way as the match_features function you already coded.
    Try to make any improvements to the matching algorithm that would speed it up.
    One suggestion is to use a space partitioning data structure like a kd-tree or some
    third party approximate nearest neighbor package to accelerate matching.
    
    Note: Doing PCA here does not count. This implementation MUST be faster than PCA
    to get credit.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                  #
    #############################################################################

    raise NotImplementedError('`accelerated_matching` function in ' +
    '`student_feature_matching.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return matches, confidences