import numpy as np
import matplotlib.pyplot as plt

from src.utils.config_loader import load_config

class SupportVectorMachine:

    def __init__(self):
        self.config = load_config()
        self.theta = None
        self.theta_0 = None

    def train(self, X, y, lam):
        X, y = X.to_numpy(), y.to_numpy()
        X = X.T

        th_opt, fs, _ = self.batch_svm_min(X, y, lam)
        self.theta = th_opt[:-1, :]
        self.theta_0 = th_opt[-1:,:]

        self.plot_loss_curve(fs)

    @staticmethod
    def _hinge(v):
        """
        Compute hinge loss element-wise.
        :param v: 1xn np array of margin values.
        :return: 1xn np array of hinge loss values; 0 if v >= 1, v-1 otherwise.
        """
        return np.where(v >= 1, 0, 1 - v)

    def _hinge_loss(self, X, y, theta, theta_0):
        """
        Compute hinge loss for full dataset
        :param X: dxn np array, d is the no. of features and n the no. of examples.
        :param y: 1xn np array of labels, n is the no. of labels.
        :param theta: dx1 weight vector.
        :param theta_0: bias term
        """
        return self._hinge(y * (theta.T @ X) + theta_0)

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
        regularization_term = lam * np.linalg.norm(theta)**2
        return np.mean(self._hinge_loss(X, y, theta, theta_0)) + regularization_term

    @staticmethod
    def d_hinge(v):
        """
        Compute hinge loss element-wise.
        :param v: 1xn np array of margin values.
        :return: 1xn np array of hinge loss values; 0 if v >= 1, v-1 otherwise.
        """
        return np.where(v >= 1, 0, -1)

    def d_hinge_loss_theta(self, X, y, theta, theta_0):
        return self.d_hinge(y * (theta.T @ X) + theta_0) * y * X

    def d_hinge_loss_theta_0(self, X, y, theta, theta_0):
        return self.d_hinge(y * (theta.T @ X) + theta_0) * y

    def d_svm_objective_theta(self, X, y, theta, theta_0, lam):
        return np.mean(self.d_hinge_loss_theta(X, y, theta, theta_0), axis = 1, keepdims = True) + lam * 2 * theta

    def d_svm_objective_theta_0(self, X, y, theta, theta_0):
        return np.mean(self.d_hinge_loss_theta_0(X, y, theta, theta_0), axis = 1, keepdims = True)

    def svm_objective_grad(self, X, y, theta, theta_0, lam):
        grad_theta = self.d_svm_objective_theta(X, y, theta, theta_0, lam)
        grad_theta_0 = self.d_svm_objective_theta_0(X, y, theta, theta_0)
        return np.vstack([grad_theta, grad_theta_0])

    def batch_svm_min(self, X, y, lam):
        epochs = self.config['EPOCHS']
        init = np.zeros((X.shape[0] + 1, 1))

        def f(th):
            return self.svm_objective(X, y, th[:-1, :], th[-1:,:], lam)

        def df(th):
            return self.svm_objective_grad(X, y, th[:-1, :], th[-1:,:], lam)

        x, fs, xs = self.gradient_descent(f, df, init, 0.01, epochs)
        return x, fs, xs

    @staticmethod
    def gradient_descent(f, df, init, step_size, epochs):
        x = init.copy()
        fs = [f(x)]
        xs = [x.copy()]
        for i in range(1, epochs + 1):
            grad = df(x)
            x = x - step_size * grad

            fs.append(f(x))
            xs.append(x.copy())

        return x, fs, xs

    @staticmethod
    def plot_loss_curve(losses, title="Training Loss Over Epochs"):
        """
        Plot the loss curve.

        Args:
            losses (list or array): Loss values recorded per epoch.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(losses, color='b')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True)
        plt.show()






