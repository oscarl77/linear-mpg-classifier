import numpy as np

from src.models.Perceptron import Perceptron
from utils.data_loader import get_test_set
from utils.config_loader import load_config

def test():
    config = load_config()
    weights = config["SAVED_MODELS"]["model.v1"]
    theta, theta_0 = np.array(weights[0]), np.array(weights[1])
    X, y = get_test_set()

    perceptron = Perceptron(epochs=0)
    perceptron.test(X, y, theta, theta_0)

if __name__ == '__main__':
    test()
