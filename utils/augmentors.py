"""Augmentors.

Defines functions that augment feature sets.

"""
import numpy as _np

from linalg import append_left as _append_left
from stats import validate_feature_set
from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError


def constant_augmentor(X):
    """Constant Linear-Combinations Augmentor.

    Adds a unit-valued column to the left of the given feature set to enable
    linear combinations with constant values.

    Args:
        X (np.matrix): Feature set. Shape: (n x d).

    Returns:
        np.matrix: Feature set capable of linear combinations with constant
            values.

    Raises:
        InvalidFeatureSetError: If feature set is somehow invalid.

    """
    validate_feature_set(X)

    v = _np.matrix(_np.ones((X.shape[0], 1)))
    """np.matrix: Unit-valued vector to append to feature set."""


    return _append_left(X, v)
