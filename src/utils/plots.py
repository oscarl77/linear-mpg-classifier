import matplotlib.pyplot as plt

from src.utils.config_loader import load_config

def plot_loss_and_accuracy(losses, accuracies):
    config = load_config()
    experiment_name = config["EXPERIMENT_NAME"]
    plt.figure(figsize=(12, 6))
    plt.suptitle(experiment_name, fontsize=16, fontweight='bold')
    plot_loss_curve(losses)
    plot_accuracy_curve(accuracies)
    plt.tight_layout()
    plt.show()

def plot_loss_curve(losses, title="Training Loss Over Epochs"):
    """
    Plot the loss curve.

    Args:
        losses (list or array): Loss values recorded per epoch.
        title (str): Title of the plot.
    """
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

def plot_accuracy_curve(accuracies, title="Training Accuracy Over Epochs"):
    """
    Plot the accuracy curve.
    :param accuracies:
    :param title:
    :return:
    """
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)