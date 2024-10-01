# Author: Harpa Gudjonsdottir
# Date: 10.09.2024
# Project: 03_linear_regression
# Acknowledgements: template file from https://github.com/T809DATA/T809DATA_2024/blob/master/

import torch
import matplotlib.pyplot as plt
import numpy as np

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    # Define variables for N, D, and M
    N, D = features.shape 
    M = mu.shape[0]  

    # Initialize the output matrix
    output_matrix = torch.zeros((N, M))

    # Set the covariance matrix
    covariance_matrix = var * torch.eye(D)
    
    # Loop over each basis function mean vector
    for k in range(M):
        # Create a multivariate normal distribution for each mean vector
        multivariate = multivariate_normal(mean=mu[k].numpy(), cov=covariance_matrix)

        # Compute the basis function values for all N data points
        output_matrix[:, k] = torch.tensor(multivariate.pdf(features.numpy()))                             
                            
    return output_matrix


def _plot_mvn(
        fi: torch.Tensor,
        M: float):
    for i in range(M):
        x_axis = fi[:, 0].numpy()
        plt.plot(x_axis, fi[:, i].numpy(), label=f'Basis Function {i+1}')


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    pass


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    pass


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # all test code can be found in tests.py
    pass