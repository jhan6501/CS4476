import numpy as np
import torch
from scipy import stats

from proj4_code.feature_matching.SIFTNet import get_siftnet_features
from proj4_code.utils import generate_sample_points


def pairwise_distances(X, Y):
    """
    This method will be very similar to the pairwise_distances() function found
    in sklearn (https://scikit-learn.org/stable/modules/generated/sklearn
    .metrics.pairwise_distances.html)
    However, you are NOT allowed to use any library functions like this
    pairwise_distances or pdist from scipy to do the calculation!

    The purpose of this method is to calculate pairwise distances between two
    sets of vectors. The distance metric we will be using is 'euclidean',
    which is the square root of the sum of squares between every value.
    (https://en.wikipedia.org/wiki/Euclidean_distance)

    Useful functions:
    -   np.linalg.norm()

    Args:
    -   X: N x d numpy array of d-dimensional features arranged along N rows
    -   Y: M x d numpy array of d-dimensional features arranged along M rows

    Returns:
    -   D: N x M numpy array where d(i, j) is the distance between row i of
    X and row j of Y
    """
    N, d_y = X.shape
    M, d_x = Y.shape
    assert d_y == d_x

    # D is the placeholder for the result
    D = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return D


def nearest_neighbor_classify(train_image_feats,
                              train_labels,
                              test_image_feats,
                              k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which can increase the
    performance.

    Useful functions:
    -   D = pairwise_distances(X, Y) computes the distance matrix D between
    all pairs of rows in X and Y. This is the method you implemented above.
        -  X is a N x d numpy array of d-dimensional features arranged along
        N rows
        -  Y is a M x d numpy array of d-dimensional features arranged along
        N rows
        -  D is a N x M numpy array where d(i, j) is the distance between
        row i of X and row j of Y
    - np.argsort()
    - scipy.stats.mode()

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality
    of the feature representation
    -   train_labels: N element list, where each entry is a string
    indicating the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality
    of the feature representation.
    -   k: the k value in kNN, indicating how many votes we need to check
    for the label

    Returns:
    -   pred_labels: M element list, where each entry is a string indicating
    the predicted category for each testing image
    """

    pred_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return pred_labels


def kmeans(feature_vectors, k, max_iter=100):
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.randint
    -   np.linalg.norm
    -   pairwise_distances - implemented above
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d)
            where N is the number of features and d is the dimensionality of
            the features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type
    int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, d)
    """

    # dummy centroids placeholder
    centroids = None
    np.random.seed(42)
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return centroids


def build_vocabulary(image_arrays, vocab_size=50, stride=20, max_iter=10):
    """
    This function will generate the vocabulary which will be further used
    for bag of words classification.

    To generate the vocab you first randomly sample features from the
    training set. Get SIFT features for the images using
    get_siftnet_features() method. This method takes as input the image
    tensor and x and y coordinates as arguments.
    Now cluster sampled features from all images using kmeans method
    implemented by you previously and return the vocabulary of words i.e.
    cluster centers.

    Points to note:
    *   To save computation time, you don't necessarily need to
    sample from all images, although it would be better to do so.
    *   Sample the descriptors from each image to save memory and
    speed up the clustering.
    *   For testing, you may experiment with larger
    stride so you just compute fewer points and check the result quickly.
    *   The default vocab_size of 50 is sufficient for you to get a
    decent accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() from utils.py for sampling interest points

    Useful note:
    - You will first need to convert each array in image_arrays into a float type array. Then convert each of these image arrays into a 4-D torch tensor by using torch.from_numpy(img_array) and then reshaping it to (1 x 1 x H x W) where H and W are the image height and width. You can use tensor views i.e torch.view to do the reshaping.

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where
    dim is the length of your SIFT descriptor. Each row is a cluster
    center/visual word.
    """

    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input
    data and the centroids, assign each of the data entry to the closest
    centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d) where N is the number of input data, and d is the dimension of it,
    given the standard SIFT descriptor, d  = 128
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
    indices = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return indices


def get_bags_of_sifts(image_arrays, vocabulary, stride=5):
    """
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet. 
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() from utils.py for sampling interest points
    -   get_siftnet_features() from SIFTNet: you can pass in the image
    tensor in grayscale, together with the sampled x and y positions to
    obtain the SIFT features
    -   np.histogram() : easy way to help you calculate for a particular
    image, how is the visual words span across the vocab. Check https://numpy.org/doc/stable/reference/generated/numpy.histogram.html for examples on how to use a histogram on an input array.
    -   np.linalg.norm() for normalizing the histogram


    Useful note:
    - You will first need to convert each array in image_arrays into a float type array. Then convert each of these image arrays into a 4-D torch tensor by using torch.from_numpy(img_array) and then reshaping it to (1 x 1 x H x W) where H and W are the image height and width. You can use tensor views i.e torch.view to do the reshaping.

    Args:
    -   image_arrays: A list of N PIL Image objects
    -   vocabulary: A numpy array of dimensions: vocab_size x 128 where each
    row is a kmeans centroid or visual word.
    -   stride: same functionality as the stride in build_vocabulary().

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
    feature representation. In this case, d will equal the number of
    clusters or equivalently the number of entries in each image's histogram
    (vocab_size) below.
    """
    # load vocabulary
    vocab = vocabulary

    vocab_size = len(vocab)
    num_images = len(image_arrays)

    feats = np.empty((num_images, vocab_size))

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return feats
