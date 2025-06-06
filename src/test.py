import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.Perceptron import Perceptron
from utils.data_loader import get_test_set
from utils.config_loader import load_config

def test():
    """Prepares and runs the testing process for the perceptron model"""
    config = load_config()
    # Import saved weights
    weights = config["SAVED_MODELS"]["model.v1"]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    X, y = get_test_set()

    perceptron = Perceptron()
    predictions = perceptron.test(X, y, theta, theta_0)

    cm = confusion_matrix(y, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    test()
