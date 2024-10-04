# Author: Harpa Gudjonsdottir
# Date: 01.10.2024
# Project: 04_backprop
# Acknowledgements: template file from https://github.com/T809DATA/T809DATA_2024/blob/master/
from typing import Union
import torch

from tools import load_iris, split_train_test


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the sigmoid of x
    """
    # To avoid overflows return 0.0 if x<-100 else return the nonlinear activation function
    return torch.where(x < -100, torch.tensor(0.0), 1 / (1 + torch.exp(-x)))


def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the derivative of the sigmoid of x.
    """
    # Calculate the sigmoid of x
    sigmoid_x = sigmoid(x)
    # Calcuate and return the derivative
    return sigmoid_x * (1 - sigmoid_x)


def perceptron(x: torch.Tensor, w: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
    """
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    """
    # Calculate the weighted sum
    weighted_sum = x @ w
    # Calculate the output of the actication function
    activation_function = sigmoid(weighted_sum)
    # Return the weighted sum and output of the activation function
    return weighted_sum, activation_function


def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    """
    # z0: Create z0 by adding 1.0 at the beginning of x to match the bias weight
    z0 = torch.cat((torch.tensor([1.0]), x), dim=0).unsqueeze(0)

    # a1, z1 without bias: Calculate the hidden layer weighted sum and output with perceptron
    a1, z1_no_bias = perceptron(z0, W1)

    # z1: Add the bias to z1
    z1 = torch.cat((torch.tensor([[1.0]]), z1_no_bias), dim=1)

    # a2, y: Calculate the output layer weighted sum and output with perceptron
    a2, y = perceptron(z1, W2)

    return y, z0, z1, a1, a2


def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    """
    # 1: Run fnn on the input
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # 2: Calculate delta_k
    delta_k = y - target_y

    # 3: Calculate delta_j
    delta_j_full = torch.matmul(delta_k, W2.T)
    # Remove the bias term
    delta_j = delta_j_full[:, 1:] * d_sigmoid(a1) 

    # 4: Initialize dE1 and dE2 as zero-matrices with the same shape as W1 and W2
    dE1 = torch.zeros_like(W1)
    dE2 = torch.zeros_like(W2)

    # 5: Calculate dE1_{i,j} and dE2_{j,k}
    dE1 = torch.matmul(z0.T, delta_j)
    dE2 = torch.matmul(z1.T, delta_k)

    return y, dE1, dE2


def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    """
    ...


def test_nn(
    X: torch.Tensor, M: int, K: int, W1: torch.Tensor, W2: torch.Tensor
) -> torch.Tensor:
    """
    Return the predictions made by a network for all features
    in the test set X.
    """
    ...


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # all test code can be found in tests.py
    pass
