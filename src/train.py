from src.config import config
from src.utils.data_loader import get_training_set
from src.models.Perceptron import Perceptron

def train():
    EPOCHS = config['EPOCHS']
    X, y = get_training_set()

    perceptron = Perceptron(EPOCHS)
    theta, theta_0 = perceptron.train(X, y)

if __name__ == "__main__":
    train()