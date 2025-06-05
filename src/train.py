import pandas as pd

from src.utils.data_loader import get_training_set

def train():
    X, y = get_training_set()


if __name__ == "__main__":
    train()