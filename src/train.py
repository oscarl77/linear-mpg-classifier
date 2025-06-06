from src.utils.config_loader import load_config, update_config
from src.utils.data_loader import get_training_set
from src.models.Perceptron import Perceptron

def train():
    """Prepares and runs the training process for the perceptron model"""
    config = load_config()
    experiment_name = config["EXPERIMENT_NAME"]
    X, y = get_training_set()

    perceptron = Perceptron()
    theta, theta_0 = perceptron.train(X, y)
    experiment_details = {experiment_name: (theta.tolist(), theta_0.tolist())}
    update_config("SAVED_MODELS", experiment_details)

if __name__ == "__main__":
    train()