# Author: Harpa Gudjonsdottir
# Date: 29.08.2024
# Project: 02_classification

from template import gen_data, mean_of_class, covar_of_class, likelihood_of_class, maximum_likelihood, predict
from tools import split_train_test
import numpy as np

def run_experiment(n, locs, scales, num_runs=10):
    '''
    Runs the experiment num_runs times and prints the accuracy.
    '''
    accuracies = []
    
    print("Accuracy results per run:")
    for i in range(num_runs):
        # Generate data
        features, targets, classes = gen_data(n, locs, scales)
        (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
        likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
        predictions = predict(likelihoods)

        # Compare predictions with the test targets to calculate the accuracy
        accuracy = np.sum(predictions == test_targets) / len(test_targets)
        # Keep results for total results
        accuracies.append(accuracy)

        print(f"  Run {i+1}: {accuracy:.2%}")
    
    print(f"Average accuracy: {np.mean(accuracies):.2%}")
    print(f"Std of the accuracy: {np.std(accuracies):.2%}\n")

if __name__ == "__main__":
    ...
    # SECTION 1 
    # Test the function to confirm that they match the sample input and outputs 
    #print(gen_data(1, [-1, 0, 1], [2, 2, 2]))
    #print(gen_data(2, [0, 2], [4, 4]))

    # Use the gen_data to create a dataset with a total of 50 samples from two normal
    # distributions: N(-1,sqrt(5)) and N(1,sqrt(5))
    #n = 50
    #locs = [-1,1]
    #scales = [np.sqrt(5), np.sqrt(5)]
    #features, targets, classes = gen_data(n, locs, scales)
    # Create a train and test set using split_train_test from tools.py with 80% train and 20% test split
    #(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    
    # SECTION 2
    # Create a plot with all my datapoints colored by class using pyplot.scatter
    # Plot training data with 0
    #plt.scatter(train_features, np.zeros_like(train_features), c='blue', marker='o')
    # Plot test data with x
    #plt.scatter(test_features, np.zeros_like(test_features), c='orange', marker='x')
    #plt.show()

    # SECTION 3
    # Test the function to confirm it matches the sample input and output
    #print(mean_of_class(train_features, train_targets, 0))
    # Also test with second class
    #print(mean_of_class(train_features, train_targets, 1))

    # SECTION 4
    # Test the function to confirm it matches the sample input and output    
    #print(covar_of_class(train_features, train_targets, 0))
    # Also test with second class
    #print(covar_of_class(train_features, train_targets, 1))

    # SECTION 5
    # Test the function to confirm it matches the sample input and output
    #class_mean = mean_of_class(train_features, train_targets, 0)
    #class_cov = covar_of_class(train_features, train_targets, 0)
    #print(likelihood_of_class(test_features[0:3], class_mean, class_cov))

    # SECTION 6
    # Test the function to confirm it matches the sample input and output
    #likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    #print(likelihoods)

    # SECTION 7
    # Test the function to confirm it matches the sample input and output
    #print(predict(likelihoods))

    # SECTION 8
    #print("--------------")
    #print("Starter datasets")

    #print("Experiment 1: Dataset with n=50, locs=[-1,1], scales=[sqrt(5), sqrt(5)]")
    #run_experiment(n=50, locs=[-1,1], scales=[np.sqrt(5), np.sqrt(5)])
    
    #print("Experiment 2: Dataset with n=50, locs=[-4,4], scales=[sqrt(2), sqrt(2)]")
    #run_experiment(n=50, locs=[-4,4], scales=[np.sqrt(2), np.sqrt(2)])


    #print("--------------")
    #print("Change the number of datapoints")

    #print("Experiment 3: Smaller dataset with n=10, locs=[-1,1], scales=[sqrt(5), sqrt(5)]")
    #run_experiment(n=10, locs=[-1,1], scales=[np.sqrt(5), np.sqrt(5)])
    
    #print("Experiment 4: Larger dataset with n=100, locs=[-1,1], scales=[sqrt(5), sqrt(5)]")
    #run_experiment(n=100, locs=[-1,1], scales=[np.sqrt(5), np.sqrt(5)])

    #print("--------------")
    #print("Change the mean")

    #print("Experiment 5: Close means, n=50, locs=[-0.5, 0.5], scales=[sqrt(2), sqrt(2)]")
    #run_experiment(n=50, locs=[-0.5, 0.5], scales=[np.sqrt(2), np.sqrt(2)])

    #print("Experiment 6: Extremely close means, n=50, locs=[0, 0.5], scales=[sqrt(2), sqrt(2)]")
    #run_experiment(n=50, locs=[0, 0.5], scales=[np.sqrt(2), np.sqrt(2)])

    #print("Experiment 7: Same means, n=50, locs=[0, 0], scales=[sqrt(2), sqrt(2)]")
    #run_experiment(n=50, locs=[0, 0], scales=[np.sqrt(2), np.sqrt(2)])

    #print("Experiment 8: Far apart means, n=50, locs=[-10, 10], scales=[sqrt(2), sqrt(2)]")
    #run_experiment(n=50, locs=[-10, 10], scales=[np.sqrt(2), np.sqrt(2)])

    #print("--------------")
    #print("Change the std")

    #print("Experiment 9: Increase std dev, n=50, locs=[-4, 4], scales=[sqrt(4), sqrt(4)]")
    #run_experiment(n=50, locs=[-4, 4], scales=[np.sqrt(4), np.sqrt(4)])

    #print("Experiment 10: Decrease std dev, n=50, locs=[-4, 4], scales=[sqrt(1), sqrt(1)]")
    #run_experiment(n=50, locs=[-4, 4], scales=[np.sqrt(1), np.sqrt(1)])

    #print("Experiment 11: Different std dev, n=50, locs=[-4, 4], scales=[sqrt(1), sqrt(3)]")
    #run_experiment(n=50, locs=[-4, 4], scales=[np.sqrt(1), np.sqrt(3)])