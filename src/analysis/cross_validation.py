import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold

from src.utils.data_loader import load_unscaled_data
from src.utils.config_loader import load_config
from src.train import train_svm
from src.test import test_svm

def cross_val_poly_degree():
    X_train, X_test, y_train, y_test = load_unscaled_data()
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    kf = KFold(n_splits=5, random_state=23, shuffle=True)
    results = {}

    config = load_config()

    degrees = [1,2,3,4]
    for degree in degrees:
        accuracies = []
        poly = PolynomialFeatures(degree)
        experiment_name = f"config_degree_{degree}"

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit poly on train, transform both train and val
            X_train_poly = poly.fit_transform(X_train)
            X_val_poly = poly.transform(X_val)

            # Scale polynomial features
            scaler = StandardScaler()
            X_train_poly_scaled = scaler.fit_transform(X_train_poly)
            X_val_poly_scaled = scaler.transform(X_val_poly)

            train_svm(X_train_poly_scaled, y_train, config, experiment_name)

            config = load_config()
            accuracy = test_svm(X_val_poly_scaled, y_val, config, experiment_name)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        results[degree] = avg_accuracy
        print(f"Degree {degree}: Average CV accuracy = {avg_accuracy:.2f}")

    best_degree = max(results, key=results.get)
    print(f"\nBest polynomial degree: {best_degree} with accuracy {results[best_degree]:.2f}")

if __name__ == '__main__':
    cross_val_poly_degree()