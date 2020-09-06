from os.path import basename

import numpy as np

import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path

    model = GDA()
    model.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
    util.plot(
        x_val, y_val, model.theta, save_path=basename(save_path).split(".")[0] + ".jpg"
    )

    yhat = model.predict(x_val)
    np.savetxt(save_path, yhat)

    print(f"GDA acc: {util.compute_accuracy(y_val, yhat)}")
    print(f"GDA log loss: {util.compute_log_loss(y_val, yhat)}")
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=0.01, max_iter=10000, eps=1e-5, theta_0=None, verbose=True
    ):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters

        m, d = x.shape

        phi = y.mean()
        mu_0 = x[y == 0].mean(axis=0, keepdims=True)
        mu_1 = x[y == 1].mean(axis=0, keepdims=True)

        sigma_0 = x[y == 0] - mu_0
        sigma_0 = sigma_0.T @ sigma_0

        sigma_1 = x[y == 1] - mu_1
        sigma_1 = sigma_1.T @ sigma_1

        sigma = (sigma_0 + sigma_1) / m
        inv_sigma = np.linalg.inv(sigma)

        self.theta = np.zeros(d + 1)
        self.theta[0] = (
            mu_0 @ inv_sigma @ mu_0.T - mu_1 @ inv_sigma @ mu_1.T
        ) / 2 + np.log(phi / (1 - phi))
        self.theta[1:] = np.squeeze(mu_1 @ inv_sigma - mu_0 @ inv_sigma)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = self.theta[1:].reshape(-1, 1)
        return np.squeeze(1 / (1 + np.exp(-(x @ theta + self.theta[0]))))
        # *** END CODE HERE


if __name__ == "__main__":
    main(
        train_path="ds1_train.csv",
        valid_path="ds1_valid.csv",
        save_path="gda_pred_1.txt",
    )

    main(
        train_path="ds2_train.csv",
        valid_path="ds2_valid.csv",
        save_path="gda_pred_2.txt",
    )
