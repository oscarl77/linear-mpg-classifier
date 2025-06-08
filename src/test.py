import numpy as np

from src.models.Perceptron import Perceptron
from src.models.SupportVectorMachine import SupportVectorMachine
from src.analysis.confusion_matrix import plot_confusion_matrix
from utils.data_loader import get_test_set
from utils.config_loader import load_config

def test():
    """Prepares and runs the testing process for the perceptron model"""
    config = load_config()
    X, y = get_test_set()

    #test_perceptron(X, y, config)
    test_svm(X, y, config)

def test_perceptron(X, y, config):
    weights = config["SAVED_PERCEPTRON_PARAMS"]["model.v1"]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    perceptron = Perceptron()
    predictions = perceptron.test(X, y, theta, theta_0)
    plot_confusion_matrix(predictions, y)

def test_svm(X, y, config):
    weights = config["SAVED_SVM_PARAMS"]["svm.v1"]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    svm = SupportVectorMachine()
    predictions = svm.test(X, y, theta, theta_0)

if __name__ == '__main__':
    test()
