import pandas as pd

from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config

_cached_dataset = None

def get_training_set():
    X_train, _, y_train, _, = _load_full_data()
    return X_train, y_train

def get_test_set():
    _, X_test, _, y_test, = _load_full_data()
    return X_test, y_test

def _load_full_data():
    global _cached_dataset
    if _cached_dataset is None:
        df = pd.read_csv('../data/auto-mpg.tsv', sep='\t')
        X_train, X_test, y_train, y_test = _preprocess_data(df)
        _cached_dataset = X_train, X_test, y_train, y_test
    return _cached_dataset

def _preprocess_data(df):
    """
    Drops emtpy rows and unwanted features, along with splitting data into
    training and test sets.
    """
    df = df.dropna()
    _select_features(df)
    X_train_raw, X_test_raw, y_train, y_test = _split_data(df)
    X_train, X_test = _normalize_data(X_train_raw, X_test_raw)

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
    # Separate discrete and continuous features as only continuous-valued
    # columns can be standardized.
    discrete_train = raw_train['origin']
    continuous_train = raw_train.drop(columns=['origin'])

    discrete_test = raw_test['origin']
    continuous_test = raw_test.drop(columns=['origin'])

    mean = continuous_train.mean()
    std = continuous_train.std()

    normalized_train = (continuous_train - mean) / std
    normalized_test = (continuous_test - mean) / std

    # Add the discrete-valued columns back to the normalized dataset
    X_train = pd.concat([normalized_train, discrete_train], axis=1)
    X_test = pd.concat([normalized_test, discrete_test], axis=1)

    return X_train, X_test

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


def _add_car_make_feature(df):
    """Add a column in the data for make of the car"""
    df['car_make'] = df['car_name'].str.split().str[0].str.lower()