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
    x: torch.Tensor, M: int, K: int, W1: torch.Tensor, W2: torch.Tensor
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
    Train the neural network by:
    1. Forward propagating the input feature through the network.
    2. Calculating the error between the prediction the network made and the actual target.
    3. Backpropagating the error through the network to adjust the weights.
    """

    # 1: Initialize necessary variables
    N = X_train.shape[0]
    Etotal = torch.zeros(iterations)
    misclassification_rate = torch.zeros(iterations)
    last_guesses = torch.zeros(N)

    # 2: Run a loop for iterations iterations.
    for i in range(iterations):
        # 3: In each iteration we will collect the gradient error matrices for each data point.
        # 3: Start by initializing dE1_total and dE2_total as zero matrices with the same shape as W1 and W2 respectively.
        dE1_total = torch.zeros_like(W1)
        dE2_total = torch.zeros_like(W2)
        # Initialize total loss, and correct predictions(for the misclassification rate) for this iteration
        total_loss = 0.0
        correct_predictions = 0

        # 4: Run a loop over all the data points in X_train.
        for j, (x, target_y) in enumerate(zip(X_train, t_train)):
            # 4: In each iteration we call backprop to get the gradient error matrices and the output values.
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2

            # 6: For the error estimation we'll use the cross-entropy error function, (Eq 5.74 in Bishop (Eq. 4.90 in old Bishop)).
            loss = -torch.sum(
                target_y * torch.log(y) + (1 - target_y) * torch.log(1 - y)
            )
            total_loss += loss.item()

            # Check for correct prediction and track last guesses
            predicted_class = torch.argmax(y)
            actual_class = torch.argmax(target_y)
            last_guesses[j] = predicted_class

            if predicted_class == actual_class:
                correct_predictions += 1

        # 5: Once we have collected the error gradient matrices for all the data points,
        # 5: we adjust the weights in W1 and W2, using W1 = W1 - eta * dE1_total / N
        # 5: where N is the number of data points in X_train (and similarly for W2).
        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        # Store total loss
        Etotal[i] = total_loss / N

        # Calculate misclassification rate
        misclassification_rate[i] = 1 - (correct_predictions / N)

    # 7: When the outer loop finishes, we return from the function
    return W1, W2, Etotal, misclassification_rate, last_guesses


def test_nn(
    X: torch.Tensor, M: int, K: int, W1: torch.Tensor, W2: torch.Tensor
) -> torch.Tensor:
    """
    Return the predictions made by a network for all features
    in the test set X.
    """
    N = X.shape[0]
    guesses = torch.zeros(N)

    # Run through all the data points in X_test
    for i, x in enumerate(X):
        # Use ffnn to guess the classification for current point
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        predicted_class = torch.argmax(y)
        guesses[i] = predicted_class

    return guesses


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # all test code can be found in tests.py
    pass
