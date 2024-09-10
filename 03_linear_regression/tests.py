# Author: Harpa Gudjonsdottir
# Date: 10.09.2024
# Project: 01_linear_regression



from template import mvn_basis, _plot_mvn, max_likelihood_linreg, linear_model
import torch
import matplotlib.pyplot as plt
from tools import load_regression_iris

if __name__ == "__main__":
    ...
    # SECTION 1 
    # Test the function to confirm that they match the sample input and outputs 
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var)

    print(fi)