"""Data normalization and preprocessing utilities."""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(data: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the data."""
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def transform(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Normalize data using a fitted scaler."""
    return scaler.transform(data)


def inverse_transform(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Denormalize data using a fitted scaler."""
    return scaler.inverse_transform(data)


def save_scaler(scaler: StandardScaler, path: str):
    """Save scaler to disk."""
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> StandardScaler:
    """Load scaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
