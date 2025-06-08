import numpy as np

from src.models.Perceptron import Perceptron
from src.models.SupportVectorMachine import SupportVectorMachine
from src.analysis.confusion_matrix import plot_confusion_matrix
from src.utils.data_loader import get_test_set
from src.utils.config_loader import load_config

def test():
    """Prepares and runs the testing process for the perceptron model"""
    config = load_config()
    X, y = get_test_set()
    experiment_name = config["EXPERIMENT_NAME"]

    #test_perceptron(X, y, config)
    test_svm(X, y, config, experiment_name)

def test_perceptron(X, y, config):
    weights = config["SAVED_PERCEPTRON_PARAMS"]["model.v1"]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    perceptron = Perceptron()
    predictions = perceptron.test(X, y, theta, theta_0)
    plot_confusion_matrix(predictions, y)

def test_svm(X, y, config, experiment_name):
    weights = config["SAVED_SVM_PARAMS"][experiment_name]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    svm = SupportVectorMachine()
    accuracy = svm.test(X, y, theta, theta_0)
    return accuracy

if __name__ == '__main__':
    test()
