import numpy as np

class Perceptron:

    def __init__(self, epochs):
        self.epochs = epochs

    def train(self, X, y):
        (d, n) = X.shape
        X, y = X.values, y.values
        theta = np.zeros(n)
        theta_0 = 0
        correct = 0

        for epoch in range(1, self.epochs):
            correct = 0
            for i in range(n):
                x_i = X[i]
                y_i = y[i]
                prediction = self.linear_classify(x_i, theta, theta_0)
                if y_i * prediction <= 0:
                    theta += y_i * x_i
                    theta_0 = theta_0 + y_i
                else:
                    correct += 1

            avg_accuracy = (correct / n) * 100
            print(f"Epoch {epoch + 1}/{self.epochs}, Running Accuracy: {avg_accuracy:.2f}")
        return theta, theta_0

    @staticmethod
    def linear_classify(x, theta, theta_0):
        return np.sign(theta.T @ x + theta_0)

    @staticmethod
    def accuracy(predictions, labels):
        return np.mean(predictions == labels)

