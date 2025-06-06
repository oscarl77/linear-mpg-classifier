import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.models.Perceptron import Perceptron
from src.utils.data_loader import get_training_set

def run_model_in_2d():
    X_train, y_train = get_training_set()
    # reduce dimensionality of data to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train)
    # train perceptron on 2D data
    perceptron = Perceptron()
    theta, theta_0 = perceptron.train(X_2d, y_train)

    # define plot limits to be centered around the data
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    # create mesh grid of many points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # model classifies all points in mesh grid using trained weight to produce
    # the decision regions.
    Z = perceptron.test_pca(np.c_[xx.ravel(), yy.ravel()], theta, theta_0)
    # As the model outputs a 1D array, we convert this 1D array into a mesh grid.
    Z = Z.reshape(xx.shape)

    # plot the decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

    # adjust plot limits to zoom in on where the data points are
    plt.xlim(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1)
    plt.ylim(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1)

    # plot original data points
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='bwr', s=20, alpha=0.6, edgecolors='k')

    # overlay decision boundary line
    w = theta
    b = theta_0
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Perceptron Decision Boundary and Regions (2D)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_model_in_2d()