import numpy as np

class SupportVectorMachine:

    def __init__(self):
        self.theta = None
        self.theta_0 = None

    def train(self, X, y, lam, learning_rate, epochs):
        X, y = X.to_numpy(), y.to_numpy()
        X = X.T
        (d, n) = X.shape
        theta = np.zeros(d)
        theta_0 = 0

        for epoch in range(epochs):
            loss = self._svm_objective(X, y, theta, theta_0, lam)

    @staticmethod
    def _hinge(v):
        """
        Compute hinge loss element-wise.
        :param v: 1xn np array of margin values.
        :return: 1xn np array of hinge loss values; 0 if v >= 1, v-1 otherwise.
        """
        return np.where(v >= 1, 0, 1-v)

    def _hinge_loss(self, X, y, theta, theta_0):
        """
        Compute hinge loss for full dataset
        :param X: dxn np array, d is the no. of features and n the no. of examples.
        :param y: 1xn np array of labels, n is the no. of labels.
        :param theta: dx1 weight vector.
        :param theta_0: bias term
        """
        return self._hinge(y * (theta.T @ X) + theta_0)

    def _svm_objective(self, X, y, theta, theta_0, lam):
        """
        Compute the SVM objective function on the full dataset
        :param X: dxn np array, d is the no. of features and n the no. of examples.
        :param y: 1xn np array of labels, n is the no. of labels.
        :param theta: dx1 weight vector.
        :param theta_0: bias term.
        :param lam: regularization term.
        :return: mean hinge loss + L2 regularization term.
        """
        regularization_term = lam * np.linalg.norm(theta)**2
        return np.mean(self._hinge_loss(X, y, theta, theta_0)) + regularization_term

