"""
    Python version: 3.11.1
         Rotem Halbreich & Daniel Tzafrir
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mode
from tabulate import tabulate
import math


'''
A method which calculates the Lp distance
'''
def lp_dist(x1, x2, p):
    if p == 1:
        return sum([abs(x1[i] - x2[i]) for i in range(len(x1))])
    elif p == 2:
        return math.sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1))]))
    elif p == np.inf:
        return max([abs(x1[i] - x2[i]) for i in range(len(x1))])
    else:
        raise ValueError("Invalid norm")


'''
The implementation of K-NN algorithm
'''
def knn(X_train, y_train, x_test, k, p):
    distances = [lp_dist(x_test, X_train[i], p) for i in range(len(X_train))]
    k_neighbors = np.argsort(distances)[:k]
    k_neighbors_labels = [y_train[i] for i in k_neighbors]
    return mode(k_neighbors_labels)


'''
Read data for dataset "haberman.data"
'''
def read_data(file_name):
    data = np.genfromtxt(file_name, delimiter=',', dtype=str, encoding=None, skip_footer=0)
    X, y = data[:, :3], data[:, -1]
    return X.astype(float), y.astype(float)


'''
Read data for dataset "squares.txt"
'''
# def read_data(file_name):
#     data = np.genfromtxt(file_name, dtype=float, encoding=None, skip_footer=0)
#     X, y = data[:, :2], data[:, -1]
#     return X.astype(float), y.astype(float)


'''
A method used for printing a table of {K, P, true_errors, empirical_errors & the difference between both errors}
'''
def print_table(K, P, true_errors, empirical_errors):
    print("Empirical & True Errors averaged over 100 Knn runs:")
    rows = []
    for (i, j), _ in np.ndenumerate(true_errors):
        rows.append([f'P = {str(P[i])}:', f'K = {str(K[j])}', "{:.2f}%".format(true_errors[i][j]),
                     "{:.2f}%".format(empirical_errors[i][j]),
                     "{:.2f}%".format(true_errors[i][j] - empirical_errors[i][j])])
    table = tabulate(rows, headers=['P:', 'K:', 'True Error Average:', 'Empirical Error Average:', 'Difference:'],
                     tablefmt='fancy_grid')
    print(table)


'''
Plotting graphs for {true_errors, empirical_errors & the difference between both errors}
'''
def plot_results(k_values, p_values, true_errors, empirical_errors):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # create a figure with 3 subplots
    axs[0].set_title("True Error", fontweight='bold')
    axs[0].set_xlabel("K Neighbors")
    axs[0].set_ylabel("Prediction Error")
    for i, p in enumerate(p_values):
        axs[0].plot(k_values, true_errors[i], linewidth=3, label=f"p={p}")
    axs[0].legend()

    axs[1].set_title("Empirical Error", fontweight='bold')
    axs[1].set_xlabel("K Neighbors")
    axs[1].set_ylabel("Prediction Error")
    for i, p in enumerate(p_values):
        axs[1].plot(k_values, empirical_errors[i], linewidth=3, label=f"p={p}")
    axs[1].legend()

    axs[2].set_title("Difference", fontweight='bold')  # create a new subplot for the difference
    axs[2].set_xlabel("K Neighbors")
    axs[2].set_ylabel("Prediction Error")
    for i, p in enumerate(p_values):
        diff = true_errors[i] - empirical_errors[i]  # calculate the difference
        axs[2].plot(k_values, diff, linewidth=3, label=f"p={p}")  # plot the difference
    axs[2].legend()

    plt.show()


'''
Plotting bar graphs for {true_errors, empirical_errors & the difference between both errors}
'''
def plot_bars(K, P, data, title):
    bar_p1 = np.arange(len(K))
    bar_p2 = [x + 0.25 for x in bar_p1]
    bar_p_inf = [x + 0.25 for x in bar_p2]
    bars = np.array([bar_p1, bar_p2, bar_p_inf])
    [plt.bar(bars[i], data[i, :], width=0.25, edgecolor='black', label='P = ' + str(p))
     for i, p in enumerate(P)]
    plt.xlabel('K Neighbors', fontsize=14)
    plt.ylabel('Prediction Error', fontsize=14)
    plt.xticks([x + 0.25 for x in range(len(K))], [str(k) for k in K])
    plt.title(title, fontweight='bold', fontsize=25)
    plt.legend()
    plt.show()


def main():
    X, y = read_data('haberman.data')
    # X, y = read_data('squares.txt')
    epochs = 100
    K = [1, 3, 5, 7, 9]
    P = [1, 2, np.inf]
    true_errors = np.zeros(shape=(len(P), len(K)))
    empirical_errors = np.zeros(shape=(len(P), len(K)))

    for i in range(epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
        print("Iteration: ", i)
        for j, k in enumerate(K):
            for l, p in enumerate(P):
                y_pred = [knn(X_train, y_train, x_test, k, p) for x_test in X_test]
                true_errors[l, j] += sum(y_pred != y_test) / len(y_test)

                y_pred_train = [knn(X_train, y_train, x_test, k, p) for x_test in X_train]
                empirical_errors[l, j] += sum(y_pred_train != y_train) / len(y_train)

    print_table(K, P, true_errors, empirical_errors)

    true_errors /= 100
    empirical_errors /= 100
    print("True Errors:")
    print(true_errors)
    print("Empirical Errors:")
    print(empirical_errors)
    print("Differences:")
    print(true_errors - empirical_errors)
    plot_results(K, P, true_errors, empirical_errors)

    plot_bars(K, P, true_errors, "True Error")
    plot_bars(K, P, empirical_errors, "Empirical Error")
    plot_bars(K, P, true_errors - empirical_errors, "Difference")


if __name__ == '__main__':
    main()


