# Important note: you do not have to modify this file for your homework.

import numpy as np

from .util import load_csv


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1.0 / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, learning_rate=0.1):
    theta = np.zeros(X.shape[1])

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print(f"Finished {i} iterations, grad {grad}, theta {theta}")
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print(f"Converged in {i} iterations, theta {theta}")
            break

        if (i + 1) % (10 * 10000) == 0:
            break

    return theta


def main():
    print("==== Training model on data set A ====")
    Xa, Ya = load_csv("ds1_a.csv", add_intercept=True)
    logistic_regression(Xa, Ya)

    print("\n==== Training model on data set B ====")
    Xb, Yb = load_csv("ds1_b.csv", add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == "__main__":
    main()
