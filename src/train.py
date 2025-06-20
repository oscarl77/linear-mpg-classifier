from src.utils.config_loader import load_config, update_config
from src.utils.data_loader import get_training_set
from src.models.Perceptron import Perceptron
from src.models.SupportVectorMachine import SupportVectorMachine
from src.utils.plots import plot_loss_and_accuracy

def train():
    """Prepares and runs the training process for the perceptron model"""
    config = load_config()
    experiment_name = config["EXPERIMENT_NAME"]
    X, y = get_training_set()

    #train_perceptron(X, y, experiment_name)
    train_svm(X, y, config, experiment_name)

def train_perceptron(X, y, experiment_name):
    perceptron = Perceptron()
    theta, theta_0 = perceptron.train(X, y)
    experiment_details = {experiment_name: (theta.tolist(), theta_0.tolist())}
    update_config("SAVED_PERCEPTRON_PARAMS", experiment_details)

def train_svm(X, y, config, experiment_name):
    epochs = config["EPOCHS"]
    regularization = config["REGULARIZATION"]
    learning_rate = config["LEARNING_RATE"]
    svm = SupportVectorMachine()
    theta, theta_0, losses, accuracies = svm.train(X, y, regularization, learning_rate, epochs)
    plot_loss_and_accuracy(losses, accuracies)
    experiment_details = {experiment_name: (theta.tolist(), theta_0.tolist())}
    update_config("SAVED_SVM_PARAMS", experiment_details)

if __name__ == "__main__":
    train()