# Author: Harpa Gudjonsdottir
# Date: 29.08.2024
# Project: 01_sequential_estimation
# Acknowledgements: template file from https://github.com/T809DATA/T809DATA_2024/blob/master/


import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    std: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    # Create the k x k identity matrix
    identity_matrix = np.eye(k)
    # Scale the identity matrix by the variance (std^2)
    covariance_matrix = (std ** 2) * identity_matrix

    # Generate the array X
    X = np.random.multivariate_normal(mean, covariance_matrix, n)
    # Return the generated array  
    return X

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    # Calculate and return
    return mu + (x - mu) / n


def _plot_sequence_estimate():
    # Generate 100 2-dimensional points with mean [0,0] and variance 3
    np.random.seed(1234)
    data = gen_data(100, 2, np.array([0, 0]), 3)

    # Set the initial estimate as (0,0)
    estimates = [np.array([0, 0])]

    # Perform update_sequence_mean for each point in the set
    for i in range(data.shape[0]):
        # Update then append the sequence mean using the current mean estimate and new data point
        estimates.append(update_sequence_mean(estimates[-1], data[i], i + 1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def _plot_mean_square_error():
    # Generate 100 2-dimensional points with mean [0, 0] and variance 3
    np.random.seed(1234)
    data = gen_data(100, 2, np.array([0, 0]), 3)

    # Set the initial estimate as [0, 0]
    estimates = [np.array([0, 0])]
    # Setup an array that captures the squared error
    squared_errors = []

    # Perform update_sequence_mean for each point in the set
    for i in range(data.shape[0]):
        # Update the sequence mean using the current mean estimate and new data point
        new_estimate = update_sequence_mean(estimates[-1], data[i], i + 1)
        # Appent the results
        estimates.append(new_estimate)

        # Calculate squared error and store it
        squared_errors.append(_square_error(np.array([0, 0]), new_estimate))

    # Plot the squared error between the estimate and the acutal mean
    plt.plot(squared_errors)
    plt.show()


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # SECTION 1 
    # Set the random seed
    #np.random.seed(1234)

    # Test the function to confirm that they match the sample input and outputs 
    #print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    #print(gen_data(5, 1, np.array([0.5]), 0.5))

    # SECTION 2
    # Create data to work with
    #section2_data = gen_data(300, 2, np.array([-1,2]), 4)

    # Visualize the data using scatter plot
    #scatter_2d_data(section2_data)

    # Visualize the data using histogram
    #bar_per_axis(section2_data)

    # SECTION 3
    #Set the random seed
    #np.random.seed(1234)

    # Re=use the same sample data as in section 2, lets set it as X to match the readme
    #X = section2_data
    #mean = np.mean(X, 0)
    #new_x = gen_data(1, 2, np.array([0, 0]), 1)
    #print(update_sequence_mean(mean, new_x, X.shape[0]+1))

    # SECTION 4
    #_plot_sequence_estimate()

    # SECTION 5
    #_plot_mean_square_error()


    
