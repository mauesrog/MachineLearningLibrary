"""Statistical Utilities.

Provides the implementation for commponly used routines in ML.

Attributes:
    See `config.utils`.

"""
import numpy as _np
from random import shuffle as _shuffle

from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidObservationSetError as _InvalidObservationSetError


def batches(X, Y, k):
    """K-Batch Creator.

    Partitions the given sets of features and observations into batches of at
    least `k` elements. The number of data points does not differ by more than
    one data point from one batch to the other.

    Args:
        X (np.matrix): Feature set. Shape: n x d.
        Y (np.matrix): Observation set. Shape: n x 1.
        k (int): Minimum number of data points per batch.

    Raises:
        AttributeError: If `X` is not a valid matrix.
        ValueError: If `X` or `Y` are empty matrices or if `k` is not a natural
            number.
        TypeError: If `X` or `Y` are not valid matrices or if `k` is not an int.

    Returns:
        list of np.matrix: Partitions of at least `k` elements per batch.

    """
    validate_datasets(X, Y)

    if type(k) != int:
        raise TypeError("Expected an 'int' for `k`, saw '%s' instead." %
                        type(k).__name__)

    if k <= 0:
        raise ValueError("Value of `k` not greater than 0: %d." % k)

    n, d = X.shape
    """(int, int): Number of data points and number of features."""

    indices = [i for i in range(0, n)]
    """list of int: Shuffled indiced of data points."""

    _shuffle(indices)

    batches = []
    """list of np.matrix: All batches."""
    n_training_points = int(_np.floor(float(n) / float(min(n, k))))
    """int: Number of data points destined for training."""
    i = None
    """int: Current data point index."""

    for q in range(0, n_training_points):
        tot = min(len(indices), k)
        """int: Number of data points to add to current batch."""
        batch = _np.zeros((tot, d + 1))
        """int: Current batch."""

        for j in range(0, tot):
            i = indices.pop()
            batch[j, :] = _np.concatenate((X[i, :], Y[i, :]), 1)

        batches.append(_np.matrix(batch))

    j = 0
    """int: Current batch offset."""

    while len(indices) > 0:
        i = indices.pop()

        datapoint = _np.matrix(_np.concatenate((X[i, :], Y[i, :]), 1))
        """np.matrix: Remaining data point."""
        m = j % len(batches)
        """int: Current batch index."""

        batches[m] = _np.matrix(_np.concatenate((batches[m], datapoint)))
        j += 1

    return batches

def normalize(X):
    """Feature Normalizer.

    Given a set of features, standardizes each data point so that it has a mean
    of 0.0 and a standard deviation of 1.0.

    Args:
        X (np.matrix): Feature set. Shape: n x d.

    Raises:
        TypeError: If `X` is not a valid matrix.
        ValueError: If `X` is an empty matrix.

    Returns:
        np.matrix: Standardized set of features.

    """
    validate_feature_set(X)

    return (X - _np.matrix(_np.mean(X, 1))) / _np.matrix(_np.std(X, 1))

def partition_data(X, Y, f):
    """Data Partitioner.

    Partition the given dataset into training and testing partitions according
    to a specific ratio.

    Args:
        X (np.matrix): Feature set. Shape: n x d.
        Y (np.matrix): Observation set. Shape: n x 1.
        f (float): Partition ratio.

    Raises:
        AttributeError: If `X` is not a valid matrix.
        TypeError: If `Y` is not a valid matrix.
        ValueError: If `f` is not a float in the range (0, 1), if `Y` is not a
            matrix of shape n x 1, or if `X` or `Y` are empty matrices.

    Returns:
        (np.matrix, np.matrix, np.matrix, np.matrix): The training sets of
            features and observations along with the testing sets of features
            and observations, respectively.

    """
    validate_datasets(X, Y, operation="partitioning")

    if type(f) != int and type(f) != float:
        raise TypeError("Expected 'float' or 'int' for `f`, saw '%s' instead." %
                        type(f).__name__)

    if f < 0.0 or f > 1.0:
        raise ValueError("Partition has to be a float in the range (0, 1), not "
                         "'%f'." % f)

    if f == 0.0:
        raise ValueError("Training set cannot be empty!")

    if f == 1.0:
        raise ValueError("Testing set cannot be empty!")

    n, d = X.shape
    """(int, int): Number of data points and number of features."""

    k = int(_np.floor(n * f))
    """int: Number of training data points."""

    train_X = _np.zeros((k, d))
    """np.matrix: Training feature set."""
    train_Y = _np.zeros((k, 1))
    """np.matrix: Training observation set."""

    test_X = _np.zeros((n - k, d))
    """np.matrix: Testing feature set."""
    test_Y = _np.zeros((n - k, 1))
    """np.matrix: Testing observation set."""

    indices = [i for i in range(0, n)]
    """list of int: Shuffled indiced of data points."""

    _shuffle(indices)

    i = None
    """int: Current data point index."""

    # Fill up training data.
    for j in range(0, k):
        i = indices.pop()

        train_X[j, :] = X[i, :]
        train_Y[j, :] = Y[i, :]

    # Fill up testing data.
    for j in range(0, n - k):
        i = indices.pop()

        test_X[j, :] = X[i, :]
        test_Y[j, :] = Y[i, :]

    return _np.matrix(train_X), _np.matrix(train_Y), _np.matrix(test_X), \
           _np.matrix(test_Y)

def validate_datasets(X, Y, **kwargs):
    """Dataset Validator.

    Validates the given feature set and observation set for compatibility,
    type agreement, and non-triviality.

    Args:
        X (np.matrix): Feature set. Shape: n x d.
        Y (np.matrix): Observation set. Shape: n x 1.
        **kwargs: Optinally includes description of exception in case of dataset
            incompatibilty.

    Raises:
        IncompatibleDataSetsError: If the numbers of rows in `X` and `Y` do not
            match.
        InvalidFeatureSetError: If `X` is not a non-trivial matrix.
        InvalidObservationSetError: If `X` is not a non-trivial matrix.

    """
    validate_feature_set(X)
    validate_observation_set(Y)

    if X.shape[0] != Y.shape[0]:
        raise _IncompatibleDataSetsError(X, Y, **kwargs)

def validate_feature_set(X):
    """Dataset Validator.

    Validates the given feature set for type and non-triviality.

    Args:
        X (np.matrix): Feature set. Shape: n x d.

    Raises:
        InvalidFeatureSetError: If `X` is an invalid or trivial matrix.

    """
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if X.size == 0:
        raise _InvalidFeatureSetError(X, reason="Received empty matrix.")

def validate_observation_set(Y):
    """Dataset Validator.

    Validates the given observation set for type and non-triviality.

    Args:
        Y (np.matrix): Observation set. Shape: n x 1.

    Raises:
        InvalidFeatureSetError: If `Y` is an invalid or trivial matrix or if
            `Y` does not have exactly one column.

    """
    if type(Y) != _np.matrix:
        raise _InvalidObservationSetError(Y, isType=True)

    if Y.size == 0:
        raise _InvalidObservationSetError(Y, reason="Received empty matrix.")

    if Y.shape[1] != 1:
        raise _InvalidObservationSetError(Y, reason="Not a column vector.")
