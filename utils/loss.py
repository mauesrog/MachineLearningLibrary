"""Loss Functions.

Defines different methods to evaluate predictions against real observations.

"""
import numpy as _np

from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError
from stats import validate_observation_set

def mse(Y, Y_hat):
    """Mean Squared Error (MSE).

    Computes the MSE of predicted observations against real ones.

    Args:
        Y (np.matrix): Observation set. Shape: n x 1.
        Y_hat (np.matrix): Predicted observations. Shape: n x 1.

    Returns:
        float: Mean squared error.

    """
    validate_observation_set(Y)
    validate_observation_set(Y_hat)

    if Y.shape != Y_hat.shape:
        raise _IncompatibleDataSetsError(Y, Y_hat, "subtraction")

    n = Y.shape[0]
    """int: Number of observations."""

    return _np.linalg.norm(Y - Y_hat) ** 2.0 / n
