import numpy as np

from src.utils.config_loader import load_config

class Perceptron:
    """
    Implementation of a simple Perceptron learning algorithm fpr binary classification.

    This model learns a hyperplane to separate car data into good or bad mpg.

    Achieved a test accuracy of 90.82%
    """

    def test(self, X, y, theta, theta_0):
        """
        Algorithm to run a trained perceptron on the test dataset.
        :param X: d x n matrix where d is the no. of features and n the no. of examples.
        :param y: n-dimensional vector where n is the no. of labels.
        :param theta: trained weight vector of dimension d where d is the no. of features.
        :param theta_0: trained bias term.
        :return: Numpy array of predicted values.
        """
        X, y = X.to_numpy(), y.to_numpy()
        X = X.T
        (d, n) = X.shape
        correct = 0
        predictions = []
        for i in range(n):
            # Iterate through each data point and corresponding label.
            x_i = X[:, i]
            y_i = y[i]
            prediction = self.linear_classify(x_i, theta, theta_0)
            predictions.append(prediction)
            if prediction == y_i:
                correct += 1
        test_accuracy = (correct / n) * 100
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        return np.array(predictions)

    def train(self, X, y):
        """
        Algorithm to train the perceptron on the training dataset.
        :param X: d x n matrix where d is the no. of features and n the no. of examples.
        :param y: n-dimensional vector where n is the no. of labels.
        :return: theta, theta_0: trained weight vector and bias term.
        """
        config = load_config()
        EPOCHS = config["EPOCHS"]
        # Convert dataset and labels to numpy arrays if not already done.
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        X = X.T
        (d, n) = X.shape
        # Initialize weight and bias to 0
        theta = np.zeros(d)
        theta_0 = 0

        for epoch in range(1, EPOCHS):
            correct = 0
            for i in range(n):
                # Iterate through each datapoint and corresponding label.
                x_i = X[:, i]
                y_i = y[i]
                prediction = self.linear_classify(x_i, theta, theta_0)
                # Define the update rule for the perceptron.
                if y_i * prediction <= 0:
                    # We update the weight and bias if the perceptron
                    # mis-classifies a datapoint.
                    theta += y_i * x_i
                    theta_0 = theta_0 + y_i
                else:
                    correct += 1

            avg_accuracy = (correct / n) * 100
            print(f"Epoch {epoch}, Accuracy: {avg_accuracy:.2f}%")
        return theta, theta_0

    def test_pca(self, X, theta, theta_0):
        """
        Test algorithm specifically for PCA.
        :param X: Mesh grid of points.
        :param theta: trained weight vector.
        :param theta_0: trained bias scalar.
        :return: Numpy array of predicted values.
        """
        (n, d) = X.shape
        predictions = []
        for i in range(n):
            x_i = X[i]
            prediction = self.linear_classify(x_i, theta, theta_0)
            predictions.append(prediction)
        return np.array(predictions)

    @staticmethod
    def linear_classify(x, theta, theta_0):
        """
        The hypothesis class that classifies a datapoint.
        :param x: d-dimensional feature vector where d is the no. of features.
        :param theta: d-dimensional weight vector where d is the no. of features.
        :param theta_0: The bias term.
        :return: predicted class label, +1 or -1
        """
        return np.sign(theta.T @ x + theta_0)

    @staticmethod
    def accuracy(predictions, labels):
        """Calculate the accuracy of perceptron predictions.
        :param predictions: Numpy array of predicted values.
        :param labels: n-dimensional vector where n is the no. of labels.
        :return: the proportion of correct predictions.
        """
        return np.mean(predictions == labels)

