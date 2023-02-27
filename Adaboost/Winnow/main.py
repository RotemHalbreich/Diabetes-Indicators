import numpy as np

"""
Rotem Halbreich & Daniel Tzafrir

Winnow Algorithm
"""
def winnow(dataset):
    # Number of features
    n = len(dataset[0]) - 1

    # initialization of weights & errors
    weights = [1] * n
    errors = 0

    i = 0
    # For each entry t in the training data set
    while i < len(dataset):

        # The label for the Xi
        label = dataset[i][-1]

        # Sigma{wj * Fj(Xi)}
        sigma = 0

        # iterate over every feature excluding label
        for j in range(n):
            sigma += weights[j] * dataset[i][j]

        guess = 0
        if sigma >= n:
            guess = 1

        # If we made a mistake, update weight vector:
        # if we guessed "+" but "-" was true
        if guess == 1 and label == 0:
            errors += 1
            for j in range(n):
                if dataset[i][j] == 1:
                    weights[j] = 0

            i = -1

        # if we guessed "-" but "+" was true
        elif guess == 0 and label == 1:
            errors += 1
            for j in range(n):
                if dataset[i][j] == 1:
                    weights[j] *= 2

            i = -1
        i += 1
    print('Weights: ', weights)
    return errors


if __name__ == '__main__':
    filename = 'winnow_vectors.txt'
    matrix = np.loadtxt(filename)
    err = winnow(matrix)
    print('Number of errors: ', err)
