import numpy as np

import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set
    # Use np.savetxt to save predictions on eval set to save_path

    model = LogisticRegression()
    model.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_val, y_val, model.theta, save_path="db.jpg")

    np.savetxt(save_path, model.predict(x_val))

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=0.01, max_iter=1000000, eps=1e-5, theta_0=None, verbose=True
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        m, d = x.shape
        y = y[:, np.newaxis]

        if self.theta is None:
            self.theta = np.zeros((1, d), dtype=np.float64)
        else:
            self.theta = np.asarray(self.theta)
            assert self.theta.shape == (1, d)

        for i in range(self.max_iter):
            prev = self.theta

            yhat = self.predict(x)[:, np.newaxis]

            if self.verbose:
                loss = self._compute_loss(y, yhat)
                print(f"Loss step {i} : {loss:.4f}")

            grad = -(x * (y - yhat)).mean(axis=0, keepdims=True).T
            H = np.identity(m, dtype=np.float64) * yhat * (1 - yhat)
            H = x.T @ H @ x

            self.theta -= np.pinv(H, hermitian=True) @ grad

            if (self.theta - prev).abs().sum() < self.eps:
                break

        # *** END CODE HERE ***

    def _compute_loss(self, y, yhat):
        return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return x @ self.theta.T
        # *** END CODE HERE ***


if __name__ == "__main__":
    main(
        train_path="ds1_train.csv",
        valid_path="ds1_valid.csv",
        save_path="logreg_pred_1.txt",
    )

    main(
        train_path="ds2_train.csv",
        valid_path="ds2_valid.csv",
        save_path="logreg_pred_2.txt",
    )
