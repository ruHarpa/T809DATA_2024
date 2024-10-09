# Author: Harpa Gudjonsdottir
# Date: 01.10.2024
# Project: 04_backprop


from template import sigmoid, d_sigmoid, perceptron, ffnn, backprop, train_nn, test_nn
from typing import Union
import torch
from tools import load_iris, split_train_test
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == "__main__":
    ...
    # SECTION 1.1 - The Sigmoid function
    # Test the function to confirm that it matches the sample input and outputs
    #print(sigmoid(torch.Tensor([0.5])))
    # Check for overflow problems
    #print(sigmoid(torch.Tensor([-101])))
    # Test the d_sigmoid to confirm that it matches the sample input and output
    #print(d_sigmoid(torch.Tensor([0.2])))

    # SECTION 1.2 - The Perceptron function
    # Test the function to confirm that it matches the sample input and outputs
    #print(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1])))
    #print(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4])))

    # SECTION 1.3 - Forward Propagation
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(
        features, targets
    )

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
    #print("y=", y)
    #print("z0=", z0)
    #print("z1=", z1)
    #print("a1=", a1)
    #print("a2=", a2)

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
    #print("y=", y)
    #print("dE1=", dE1)
    #print("dE2=", dE2)

    # SECTOIN 2.1 - train_nn and
    # initialize the random seed to get predictable results
    torch.manual_seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    # Test the function to confirm that it matches the sample input and outputs
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1
    )
    #print("W1tr=", W1tr)
    #print("W2tr=", W2tr)
    #print("misclassification_rate=", misclassification_rate)
    #print("last_guesses=", last_guesses)

    
    # SECTION 2.2 and 2.3 - test_nn andTrain the network and test it on the Iris dataset
    # 1. Initialize the random seed to get predictable results
    manual_seed = 1234
    torch.manual_seed(manual_seed)

    # 2. Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 3. One-hot encode the target labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = one_hot_encoder.fit_transform(
        y.reshape(-1, 1)
    )

    # 4. Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=manual_seed
    )

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 5. Define the network parameters
    K = 3  # number of classes
    M = 6
    D = X_train.shape[1]
    
    # 6. Initialize weights randomly
    W1 = 2 * torch.rand(D + 1, M) - 1  
    W2 = 2 * torch.rand(M + 1, K) - 1 

    # 7. Train the network
    iterations = 500
    eta = 0.1
    W1tr, W2tr, E_total, misclassification_rate, _ = train_nn(
        X_train, y_train, M, K, W1, W2, iterations, eta
    )

    # 8. Test the network and get predictions
    guesses = test_nn(X_test, M, K, W1tr, W2tr)

    # 9. Convert one-hot encoded test labels back to original labels
    y_test_labels = torch.argmax(y_test, dim=1)

    # 10. Calculate accuracy
    accuracy = accuracy_score(y_test_labels, guesses)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # 11. Confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, guesses)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix for Iris Test Data")
    plt.colorbar()

    # 12. Plot E_total as a function of iterations
    plt.figure(figsize=(8, 6))
    plt.plot(E_total.numpy())
    plt.xlabel("Iterations")
    plt.ylabel("E_total (Cross-Entropy Loss)")
    #plt.title("E_total vs Iterations")
    plt.show()

    # 13. Plot misclassification_rate as a function of iterations
    plt.figure(figsize=(8, 6))
    plt.plot(misclassification_rate.numpy())
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification Rate")
    #plt.title("Misclassification Rate vs Iterations")
    plt.show()

    # Print the seed for reproducibility
    print(f"Random Seed used: {manual_seed}")
