import numpy as np

class SupportVectorMachine:
    """
    Soft-margin SVM
    """

    @staticmethod
    def test(X, y, theta, theta_0):
        X, y = X.to_numpy(), y.to_numpy()
        X = X.T

        margin = (theta.T @ X) + theta_0
        predictions = np.sign(margin)

        accuracy = np.mean(predictions == y) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return predictions

    def train(self, X, y, lam, learning_rate, epochs):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        y = y.to_numpy()
        X = X.T

        th_opt, fs, accuracies = self.batch_svm_min(X, y, lam, learning_rate, epochs)
        theta = th_opt[:-1, :]
        theta_0 = th_opt[-1:,:]

        return theta, theta_0, fs, accuracies

    def batch_svm_min(self, X, y, lam, learning_rate, epochs):
        init = np.zeros((X.shape[0] + 1, 1))

        def f(th):
            theta, theta_0 = th[:-1, :], th[-1:,:]
            return self.svm_objective(X, y, theta, theta_0, lam)

        def df(th):
            theta, theta_0 = th[:-1, :], th[-1:, :]
            return self.svm_objective_grad(X, y, theta, theta_0, lam)

        def accuracy_fn(X, y, theta, theta_0):
            predictions = np.sign(theta.T @ X + theta_0)
            return np.mean(predictions == y) * 100

        x, fs, accs = self.gradient_descent(f, df, init, learning_rate, epochs, accuracy_fn, X, y)
        return x, fs, accs

    def svm_objective(self, X, y, theta, theta_0, lam):
        """
        Compute the SVM objective function on the full dataset.
        :param X: dxn np array, d is the no. of features and n the no. of examples.
        :param y: 1xn np array of labels, n is the no. of labels.
        :param theta: dx1 weight vector.
        :param theta_0: bias term.
        :param lam: regularization term.
        :return: mean hinge loss + L2 regularization term.
        """
        print(f"Labels shape: {y.shape}, theta shape: {(theta).shape}")
        margin = y * (theta.T @ X) + theta_0
        hinge_loss = self.hinge(margin)
        regularization_term = lam * np.linalg.norm(theta)**2
        return np.mean(hinge_loss) + regularization_term

    def svm_objective_grad(self, X, y, theta, theta_0, lam):
        margin = y * (theta.T @ X) + theta_0
        hinge_grad = self.d_hinge(margin)
        grad_theta = np.mean(hinge_grad * y * X, axis=1, keepdims=True) + 2 * lam * theta
        grad_theta_0 = np.mean(hinge_grad * y, keepdims=True)
        return np.vstack([grad_theta, grad_theta_0])

    @staticmethod
    def hinge(v):
        """
        Compute hinge loss element-wise.
        :param v: 1xn np array of margin values.
        :return: 1xn np array of hinge loss values; 0 if v >= 1, v-1 otherwise.
        """
        return np.where(v >= 1, 0, 1 - v)

    @staticmethod
    def d_hinge(v):
        """
        Compute derivative of hinge loss element-wise w.r.t v
        :param v: 1xn np array of margin values.
        :return: 1xn np array of gradient values; 0 if v >= 1, -1 otherwise.
        """
        return np.where(v >= 1, 0, -1)

    @staticmethod
    def gradient_descent(f, df, init, step_size, epochs, eval_fn, X, y):
        """
        Gradient descent algorithm.
        :param y:
        :param X:
        :param eval_fn:
        :param f: objective function
        :param df: gradient function
        :param init: initial zeroed weights
        :param step_size: learning rate
        :param epochs: no. of iterations
        :return: tuple (x: vector of final parameters,
                        fs: list of objective functions at each iteration,
                        xs: list of weight vectors at each iteration)
        """
        x = init
        fs = [f(x)]
        xs = [x]
        evals = []
        tol = 0.01
        for i in range(1, epochs + 1):
            grad = df(x)
            x = x - step_size * grad
            fs.append(f(x))
            xs.append(x)
            theta, theta_0 = x[:-1, :], x[-1:, :]
            evals.append(eval_fn(X, y, theta, theta_0))

            if abs(fs[-2] - fs[-1]) < tol:
                print(f"Converged at epoch {i}")
                break

        return x, fs, evals