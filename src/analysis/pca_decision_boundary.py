import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from src.models.Perceptron import Perceptron
from src.utils.data_loader import get_training_set

def visualize_decision_boundary():
    """Visualize decision boundary created by perceptron on dataset."""
    perceptron = Perceptron()
    X_train, y_train = get_training_set()
    X_2D = reduce_to_2D(X_train)
    mesh_x_cords, mesh_y_cords = create_mesh_grid(X_2D)
    mesh_grid = np.c_[mesh_x_cords.ravel(), mesh_y_cords.ravel()]
    decision_regions = map_decision_regions(perceptron, X_2D, y_train, mesh_grid, mesh_x_cords)
    plot_decision_regions(X_2D, mesh_x_cords, mesh_y_cords, decision_regions)
    plot_original_datapoints(X_2D, y_train)
    label_pca_plot()

def reduce_to_2D(data):
    """Reduces n-dimensional data to a 2 dimensional space."""
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

def create_mesh_grid(data_2D):
    """
    Create a 500x500 mesh grid of points to be classified by the perceptron.
    :param data_2D: 2-dimensional training data
    :return: x and y coordinates of all points in the mesh grid.
    """
    x_min, x_max = data_2D[:, 0].min() - 1, data_2D[:, 0].max() + 1
    y_min, y_max = data_2D[:, 1].min() - 1, data_2D[:, 1].max() + 1
    mesh_x_cords, mesh_y_cords = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    return mesh_x_cords, mesh_y_cords

def map_decision_regions(perceptron, X_2D, y_train, mesh_grid, mesh_x_cords):
    """
    Obtain the perceptron's predictions on every point in the mesh grid.
    :param perceptron: model used to visualize decision boundary.
    :param X_2D: 2-dimensional training data.
    :param y_train: array of training labels.
    :param mesh_grid: 500x500 grid of points.
    :param mesh_x_cords: x coordinates of all points in the mesh grid.
    :return: 500x500 array of predictions.
    """
    theta, theta_0 = perceptron.train(X_2D, y_train)
    decision_regions = perceptron.test_pca(mesh_grid, theta, theta_0)
    # Note that our model outputs a 1D array of predictions, so this must be
    # reshaped back into a mesh grid to visualize.
    return decision_regions.reshape(mesh_x_cords.shape)

def plot_decision_regions(X_2D, mesh_x_cords, mesh_y_cords, decision_regions):
    """
    Plot the decision regions.
    :param X_2D: 2-dimensional training data.
    :param mesh_x_cords: x coordinates of all points in the mesh grid.
    :param mesh_y_cords: y coordinates of all points in the mesh grid.
    :param decision_regions: 500x500 array of predictions on the mesh grid.
    """
    plt.xlim(X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1)
    plt.ylim(X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1)
    plt.contourf(mesh_x_cords, mesh_y_cords, decision_regions, alpha=0.3, cmap='bwr')

def plot_original_datapoints(X_2D, y_train):
    """
    Scatter plot the data points provided by the 2-dimensional training data.
    :param X_2D: 2-dimensional training data.
    :param y_train: array of training labels
    """
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_train, cmap='bwr', s=20, alpha=0.6, edgecolors='k')

def label_pca_plot():
    """Label axes and title of visualization plot"""
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Perceptron Decision Boundary and Regions (2D)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    visualize_decision_boundary()
