# Author: Harpa Gudjonsdottir
# Date: 29.08.2024
# Project: 02_classification
# Acknowledgements: template file from https://github.com/T809DATA/T809DATA_2024/blob/master/


from tools import load_iris, split_train_test
from help import estimate_covariance

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    # Create an array to hold the data points, classes, and targets
    data_points = []
    classes = []
    targets = []

    # Generate data points using the rvs method of the scipy.stats.norm, and updating the classes and targets
    i = 0
    for loc, scale in zip(locs, scales):
        data_points.extend(norm.rvs(loc, scale, n))
        classes.append(i)
        targets.extend([i] * n)
        i += 1
    
    # Return the data points, targets and classes
    return np.array(data_points), np.array(targets), classes


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    # Filter the features by the selected class
    class_features = features[targets == selected_class]

    # Compute and return the mean, with axis = 0 to calculate the mean for each features
    return np.mean(class_features, axis=0) 


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    # Filter the features by the selected class
    class_features = features[targets == selected_class]

    # Compute and return the covariance
    return np.cov(class_features, rowvar=False)

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    # Compute a mulitivariate normal distribution with the input mean and covariance
    normal_distribution = multivariate_normal(class_mean, class_covar)

    # Compute and return the probability density function (PDF) value for the feature
    return normal_distribution.pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        # Estimate the mean for current class 
        means.append(mean_of_class(train_features, train_targets, class_label))
        # Estimate the covariance for current class
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        class_likelihoods = []
        for j in range(len(classes)):
            # Estimate the likelihood for each test sample 
            class_likelihoods.append(likelihood_of_class(test_features[i], means[j], covs[j]))
        # Append the result for the class features
        likelihoods.append(class_likelihoods)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    # Deterine and return a class prediction using argmax
    return np.argmax(likelihoods, 1)


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # all test code can be found in tests.py