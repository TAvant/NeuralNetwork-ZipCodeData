#!/usr/bin/python

# imported libraries 
import sys
import numpy as np
from packages import nnTrainingFuncs as nn

# define main function 
def main():

    # get observation and label data
    X_data, y_data = nn.seperate_data(sys.argv[1])

    # vectorize digit labels
    y_data = nn.vectorize_digit_labels(y_data)

    # get total number of features
    num_features = X_data.shape[1]

    # initialize previous error rate to one and empty error vector
    previous_error_rate = 1
    error_vec = np.empty((num_features,1))

    # for each increasing number of features
    for x_features in range(1, num_features+1):

        # get the weights and error for current number of features
        weight1, weight2, error_rate = nn.single_hidden_train(X_data, y_data, num_features=x_features)

        # only want to keep weights of the lowest error rate
        if error_rate < previous_error_rate: 
            weight1_optimal = weight1
            weight2_optimal = weight2

        # add error to array of errors and previous error
        error_vec[x_features-1] = error_rate
        previous_error_rate = error_rate

    # save optimal weights and error vector
    np.save('Data/weight1.npy', weight1_optimal)
    np.save('Data/weight2.npy', weight2_optimal)
    np.save('Data/error.npy', error_vec)

# call main function 
if __name__ == '__main__':
    main()