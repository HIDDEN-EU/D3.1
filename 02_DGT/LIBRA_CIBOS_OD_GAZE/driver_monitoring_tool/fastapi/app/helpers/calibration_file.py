"""Loading and validation of the pickled 3x3 calibration (homography) matrix."""

import pickle

import numpy as np


def validate_matrix(matrix):
    """Return (matrix, None) if usable, else (None, reason)."""
    if matrix is None:
        return None, "it contains None"
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "it does not contain a numeric matrix"
    if matrix.shape != (3, 3):
        return None, f"expected a 3x3 matrix, got shape {matrix.shape}"
    if not np.isfinite(matrix).all():
        return None, "the matrix contains NaN or infinite values"
    return matrix, None


def load_calibration_matrix(path):
    """Return (matrix, None), or (None, reason) if the file is missing or unusable.

    A bad file must not crash its callers: it can be empty or half-written,
    pickled by an incompatible numpy version, or left over from an older build
    that saved NaN matrices.
    """
    try:
        with open(path, 'rb') as f:
            matrix = pickle.load(f)
    except FileNotFoundError:
        return None, f"No calibration file at {path}; run a calibration first"
    except Exception as e:  # Corrupt or incompatible file: EOFError, UnpicklingError, ModuleNotFoundError, ...
        return None, f"Calibration file {path} is unreadable: {e!r}"

    matrix, problem = validate_matrix(matrix)
    if problem:
        return None, f"Calibration file {path} ignored: {problem}"
    return matrix, None
