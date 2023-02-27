"""
    Python version: 3.11.1
         Rotem Halbreich & Daniel Tzafrir
"""
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from tabulate import tabulate
from Adaboost import *

# Creates a data frame of points (x,y) and their labels
df = pd.read_table('squares.txt', delim_whitespace=True, names=('x', 'y', 'label'))
points_df = df[['x', 'y']].copy()
labels_df = df[['label']].replace([0], -1).copy()


"""
Plot a graph for Test & Train Errors
"""
def plot_err(epochs, train_err, test_err):
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Error')
    plt.plot(epochs, test_err, color='red', linewidth=4, label='Test Error')
    plt.plot(epochs, train_err, color='green', linewidth=4, label='Train Error')
    plt.legend()
    plt.show()


def main():

    empirical_errors = []
    true_errors = []

    # epochs errors lists for plotting
    epochs_train_errors = np.zeros(8)  # initialize a list of zeros of size 8 (num epochs)
    epochs_test_errors = np.zeros(8)  # initialize a list of zeros of size 8 (num epochs)

    iterations = 50
    for i in range(iterations):
        train_err_lst, test_err_lst = adaboost(points_df, labels_df, 8)

        print(i + 1, ":", "train errors: ", train_err_lst)
        print(i + 1, ":", "test errors: ", test_err_lst, "\n")

        empirical_errors.extend(train_err_lst)
        true_errors.extend(test_err_lst)

        epochs_train_errors = [a + b for a, b in zip(epochs_train_errors, train_err_lst)]
        epochs_test_errors = [a + b for a, b in zip(epochs_test_errors, test_err_lst)]

    # Each item on the epochs errors lists is the sum of the errors in all iterations of this epoch
    # dividing each one by number of iterations to get the average
    epochs_train_errors = list(map(lambda x: x / iterations, epochs_train_errors))
    epochs_test_errors = list(map(lambda x: x / iterations, epochs_test_errors))

    print("Empirical & True Errors averaged over 50 Adaboost runs:")
    rows = []
    for i in range(8):
        rows.append([f'Rule {i + 1}:', epochs_test_errors[i], epochs_train_errors[i],
                     epochs_test_errors[i] - epochs_train_errors[i]])

    table = tabulate(rows, headers=['Rule:', 'True Error Average:', 'Empirical Error Average:', 'Difference:'],
                     tablefmt='fancy_grid')
    print(table)

    print("_____________________________________________________________________________")
    print("Empirical Error Mean: ", round(mean(empirical_errors), 4))
    print("True Error Mean: ", round(mean(true_errors), 4))
    print("_____________________________________________________________________________")

    # Plot the change on the average error along the epochs to see if there's an overfitting
    plot_err([*range(1, 9)], epochs_train_errors, epochs_test_errors)


if __name__ == '__main__':
    main()
