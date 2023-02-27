from sklearn.model_selection import train_test_split
import numpy as np
from ClassifyLine import *


def get_index(data, index, tag):
    return data.iloc[index].loc[tag]

"""
Get all classifiers for all train points 
"""
def get_classifiers(points_train):
    classifiers = []  # create an empty list of line classifiers
    num_lines = points_train.shape[0]  # number of points in train set
    # add to the list all the possible lines created from 2 points
    for i in range(num_lines - 1):
        for j in range(i + 1, num_lines):
            x_i = get_index(points_train, i, 'x')
            y_i = get_index(points_train, i, 'y')
            x_j = get_index(points_train, j, 'x')
            y_j = get_index(points_train, j, 'y')
            classifiers.append(ClassifyLine(x_i, y_i, x_j, y_j))
            classifiers.append(ClassifyLine(x_i, y_i, x_j, y_j, False))
    return classifiers

"""
Computes H(X) = sign[F(X)]
"""
def compute_h_t(t, best_rules, alpha_t, x, y):
    sigma = 0
    for i in range(t + 1):
        h_t = best_rules[i].classify(x, y)  # get the best 8 rules by lines - h_t(x)
        sigma += alpha_t[i] * h_t  # F(X) = sigma[alpha_t * h_t(x)]
    if sigma < 0:
        return -1
    elif sigma > 0:
        return 1
    else:
        return 0

"""
Computes the Empirical/True errors of the dataset
"""
def compute_error(t, best_rules, rules_weights, points, labels):
    n = points.shape[0]  # number of points
    sum_err = 0
    for i in range(n):
        indicator = 0
        x_i = get_index(points, i, 'x')
        y_i = get_index(points, i, 'y')
        label_i = get_index(labels, i, 'label')
        if compute_h_t(t, best_rules, rules_weights, x_i, y_i) != label_i:  # an error in classification
            indicator = 1
        sum_err += indicator
    return sum_err / n

"""
Adaboost algorithm 
"""
def adaboost(points_df, labels_df, k):
    # Split the data randomly into 0.5 test (T) & 0.5 train (S)
    points_train, points_test, labels_train, labels_test = train_test_split(points_df, labels_df, test_size=0.5,
                                                                            shuffle=True)
    H = get_classifiers(points_train)  # classification (sign) of rules
    best_rules = []
    rules_weights = []
    best_train_errors = []  # Empirical error list
    best_test_errors = []  # Real error list
    n = points_train.shape[0]  # number of points in the train set
    weights = [1 / n] * n  # 1: Initialize points' weights to 1/n

    # 2: Identify the k=8 most important lines h_i and their respective weights
    for t in range(k):
        min_weighted_error = np.inf
        h_t = 0
        for h in H:  # 3: Compute weighted error for each h in H
            weighted_error_h = 0
            for i in range(n):
                indicator = 0  # indicator for the event [ℎ(x_𝑖)≠y_𝑖]
                x_i = get_index(points_train, i, 'x')
                y_i = get_index(points_train, i, 'y')
                label_i = get_index(labels_train, i, 'label')
                if h.classify(x_i, y_i) != label_i:  # [ℎ(x_𝑖)≠y_𝑖]
                    indicator = 1
                weighted_error_h += weights[i] * indicator  # epsilon_t(h)
            if weighted_error_h < min_weighted_error:  # 4: Select classifier with min weighted error
                min_weighted_error = weighted_error_h
                h_t = h  # the classifier with min weighted error
        best_rules.append(h_t)

        # 5: Set classifier weight (alpha_t) based on its error
        alpha_t = 0.5 * np.log((1 - min_weighted_error) / min_weighted_error)
        rules_weights.append(alpha_t)

        # The empirical error of h_t (on the train set)
        best_train_errors.append(compute_error(t, best_rules, rules_weights, points_train, labels_train))
        # The true error of h_t (on the test set)
        best_test_errors.append(compute_error(t, best_rules, rules_weights, points_test, labels_test))

        # Update point weights - D_t+1(x_i)
        z_t = 0  # normalization factor
        for i in range(n):  # 6: Update points' weights
            x_i = get_index(points_train, i, 'x')
            y_i = get_index(points_train, i, 'y')
            label_i = get_index(labels_train, i, 'label')
            weights[i] *= np.exp(-alpha_t * h_t.classify(x_i, y_i) * label_i)
            z_t += weights[i]  # sums the weights
        weights = list(map(lambda x: x / z_t, weights))  # normalization

    return best_train_errors, best_test_errors
