# Author: Harpa Gudjonsdottir
# Date: 01.10.2024
# Project: 04_backprop



from template import sigmoid, d_sigmoid, perceptron, ffnn, backprop, train_nn, test_nn
from typing import Union
import torch
from tools import load_iris, split_train_test


if __name__ == "__main__":
    ...
    # SECTION 1.1 - The Sigmoid function
    # Test the function to confirm that it matches the sample input and outputs 
    print(sigmoid(torch.Tensor([0.5])))
    # Check for overflow problems
    print(sigmoid(torch.Tensor([-101])))
    # Test the d_sigmoid to confirm that it matches the sample input and output
    print(d_sigmoid(torch.Tensor([0.2])))

    # SECTION 1.2 - The Perceptron function
    # Test the function to confirm that it matches the sample input and outputs 
    print(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1])))
    print(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4])))

    # SECTION 1.3 - Forward Propagation
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    
    # initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point:
    x = train_features[0, :]
    K = 3  # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    # Test the function to confirm that it matches the sample input and outputs 
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    print("y=",y)
    print("z0=",z0)
    print("z1=",z1)
    print("a1=",a1)
    print("a2=",a2)

    # SECTION 1.4 - Backwards Propagation
    # initialize random generator to get predictable results
    torch.manual_seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    # Test the function to confirm that it matches the sample input and outputs
    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print("y=",y)
    print("dE1=",dE1)
    print("dE2=",dE2)