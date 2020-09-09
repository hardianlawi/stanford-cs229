import matplotlib.pyplot as plt
import numpy as np

import util


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path

    model = PoissonRegression(lr)
    model.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_val, y_val, model.theta, save_path=save_path.replace(".txt", ".png"))

    np.savetxt(save_path, model.predict(x_val))
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=1e-5, max_iter=10000000, eps=1e-5, theta_0=None, verbose=True
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        m, d = x.shape
        y = y[:, np.newaxis]

        # *** START CODE HERE ***
        theta = np.random.normal(scale=1 / np.sqrt(d), size=(1, d))

        if self.theta is not None:
            theta[0] = self.theta
        else:
            theta[0] = 0.0

        self.theta = np.squeeze(theta)

        if self.verbose:
            print(f"Loss at step 0: {self._compute_loss(x, y)}")

        for i in range(1, self.max_iter + 1):
            prev = np.copy(theta)
            yhat = self.predict(x)[:, np.newaxis]

            grad = -(-yhat * x + y * x).mean(axis=0, keepdims=True)
            assert grad.shape == theta.shape

            theta -= self.step_size * grad
            self.theta = np.squeeze(theta)

            if self.verbose:
                print(f"Loss at step {i}: {self._compute_loss(x, y)}")

            if np.abs(theta - prev).sum() < self.eps:
                print(f"stopping early bcs weights diff: {np.abs(theta - prev).sum()}")
                break

        return self
        # *** END CODE HERE ***

    def _compute_loss(self, x, y):
        yhat = self.predict(x)[:, np.newaxis]
        # Loss is negative because the normalization constant is not included.
        return -(-yhat + y * np.log(yhat)).mean()

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = self.theta.reshape(-1, 1)
        score = np.squeeze(x @ theta)
        return np.exp(score)
        # *** END CODE HERE ***


if __name__ == "__main__":
    main(
        lr=1e-5,
        train_path="train.csv",
        eval_path="valid.csv",
        save_path="poisson_pred.txt",
    )
