import numpy as np
import pandas as pd


def load_data(filepath):
    """Load CSV and return dataframe."""
    data = pd.read_csv(filepath)
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())
    data.info()
    print(data.describe())
    print("Missing values:\n", data.isnull().sum())
    return data


def split_features_target(data, target_col='target'):
    """Split dataframe into feature matrix X and target vector y."""
    y = data[target_col].values
    X = data.drop([target_col], axis=1).values
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    """Shuffle and split X, y into train and test sets."""
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    """Z-score normalize using training set statistics only."""
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0)
    feat_std[feat_std == 0] = 1          # avoid divide-by-zero
    X_train_norm = (X_train - feat_mean) / feat_std
    X_test_norm  = (X_test  - feat_mean) / feat_std
    return X_train_norm, X_test_norm, feat_mean, feat_std


def make_val_split(X_train_norm, y_train, val_frac=0.1):
    """Carve a validation set off the end of the (already normalised) training data."""
    n = X_train_norm.shape[0]
    split = int(n * (1 - val_frac))
    X_tr   = X_train_norm[:split]
    X_val  = X_train_norm[split:]
    y_tr   = y_train[:split]
    y_val  = y_train[split:]
    return X_tr, X_val, y_tr, y_val
