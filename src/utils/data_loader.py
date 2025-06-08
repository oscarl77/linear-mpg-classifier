import pandas as pd

from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config

_cached_dataset = None

def get_training_set():
    """Return the training dataset and labels"""
    X_train, _, y_train, _, = load_full_data()
    return X_train, y_train

def get_test_set():
    """Return the test dataset and labels"""
    _, X_test, _, y_test, = load_full_data()
    return X_test, y_test

def load_full_data():
    """Loads the full dataset and caches it to avoid repeated loading"""
    global _cached_dataset
    if _cached_dataset is None:
        df = pd.read_csv('../data/auto-mpg.tsv', sep='\t')
        X_train, X_test, y_train, y_test = _preprocess_data(df)
        _cached_dataset = X_train, X_test, y_train, y_test
    return _cached_dataset

def load_unscaled_data():
    df = pd.read_csv('../data/auto-mpg.tsv', sep='\t')
    return _preprocess_data(df, scaled=False)

def _preprocess_data(df, scaled=True):
    """
    Drops emtpy rows and unwanted features, along with splitting data into
    training and test sets.
    """
    df = df.dropna()
    _select_features(df)
    X_train, X_test, y_train, y_test = _split_data(df)
    if scaled:
        X_train, X_test = _normalize_data(X_train, X_test)

    return X_train, X_test, y_train, y_test

def _split_data(df):
    """Split dataset into testing and training sets"""
    config = load_config()
    SEED = config["RANDOM_SEED"]
    TEST_SPLIT = config["DATA"]["TEST_SPLIT"]
    y = df['mpg']
    X_raw = df.drop(columns=['mpg'])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SPLIT, random_state=SEED)
    return X_train_raw, X_test_raw, y_train, y_test

def _normalize_data(raw_train, raw_test):
    """Normalize data using z-standardization"""
    mean = raw_train.mean()
    std = raw_train.std()

    normalized_train = (raw_train - mean) / std
    normalized_test = (raw_test - mean) / std

    return normalized_train, normalized_test

def _select_features(df):
    """
    Select which features will be used to train and test the model
    :param df: DataFrame containing car mpg data
    """
    config = load_config()
    features = config["DATA"]["FEATURES"]
    for feature in features:
        keep = features[feature]
        if feature == 'mpg':
            continue
        if keep == "False":
            df.drop(feature, axis=1, inplace=True)