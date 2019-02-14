"""Statistical Utilities.

Provides the implementation for commponly used routines in ML.

Attributes:
    See `config.utils`.

"""
import numpy as np
from random import shuffle

import pandas as pd

from config import learner_defaults
from common.exceptions import IncompatibleDataSetsError, \
                              InvalidFeatureSetError, InvalidObservationSetError
from general import appendargs, compose

DEFAULT_MIN_FEATURE_CORRELATION = learner_defaults["min_feature_correlation"]


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

    indices = [i for i in range(n)]
    """list of int: Shuffled indiced of data points."""

    shuffle(indices)

    batches = []
    """list of np.matrix: All batches."""
    n_training_points = compose(int,
                                np.floor)(float(n) / compose(float, min)(n, k))
    """int: Number of data points destined for training."""
    i = None
    """int: Current data point index."""

    for q in range(n_training_points):
        tot = compose(appendargs(min, k), len)(indices)
        """int: Number of data points to add to current batch."""
        batch = np.zeros((tot, d + 1))
        """int: Current batch."""

        for j in range(tot):
            i = indices.pop()
            batch[j, :] = np.concatenate((X[i, :], Y[i, :]), 1)

        compose(batches.append, np.matrix)(batch)

    if len(batches) == 1:
        n_left = len(indices)

        if n_left == 0:
            raise ValueError("Unable to partition %d data points into length "
                             "%d batches." % (n, k))

        batch = np.zeros((n_left, d + 1))
        """int: Current batch."""

        batch = np.concatenate([np.concatenate((X[i, :], Y[i, :]), 1)
                                for i in indices], 0)
        compose(batches.append, np.matrix)(batch)
    else:
        j = 0
        """int: Current batch offset."""

        while len(indices) > 0:
            i = indices.pop()

            datapoint = compose(np.matrix,
                                np.concatenate)((X[i, :], Y[i, :]), 1)
            """np.matrix: Remaining data point."""
            m = j % len(batches)
            """int: Current batch index."""

            batches[m] = compose(np.matrix,
                                 np.concatenate)((batches[m], datapoint))
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

    return (X - np.matrix(np.mean(X, 1))) / np.matrix(np.std(X, 1))

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

    k = int(np.floor(n * f))
    """int: Number of training data points."""

    train_X = np.zeros((k, d))
    """np.matrix: Training feature set."""
    train_Y = np.zeros((k, 1))
    """np.matrix: Training observation set."""

    test_X = np.zeros((n - k, d))
    """np.matrix: Testing feature set."""
    test_Y = np.zeros((n - k, 1))
    """np.matrix: Testing observation set."""

    indices = [i for i in range(n)]
    """list of int: Shuffled indiced of data points."""

    shuffle(indices)

    i = None
    """int: Current data point index."""

    # Fill up training data.
    for j in range(k):
        i = indices.pop()

        train_X[j, :] = X[i, :]
        train_Y[j, :] = Y[i, :]

    # Fill up testing data.
    for j in range(n - k):
        i = indices.pop()

        test_X[j, :] = X[i, :]
        test_Y[j, :] = Y[i, :]

    return np.matrix(train_X), np.matrix(train_Y), np.matrix(test_X), \
           np.matrix(test_Y)

def reduce_dimensions(X, Y, min=DEFAULT_MIN_FEATURE_CORRELATION, names=None):
    """Data Point Dimensionality Reducer.

    Args:
        X (np.matrix): Feature set. Shape: n x d.
        Y (np.matrix): Observation set. Shape: n x 1.
        min (float, optional): Determines the minimum correlation value for a
            feature to be considered relevant. Defaults to
            `DEFAULT_MIN_FEATURE_CORRELATION`.
        names (list of str): Feature names. Defaults to `None`.

    Returns:
        Reduced feature set `np.matrix` if no feature names are provided, a
            tuple with the reduced feature set and feature names otherwise.

    Raises:
        ValueError: If no features have a correlation to `Y` greater than or
            equal to `min`.

    Todo:
        Get rid of `pandas`.

    """
    validate_datasets(X, Y)

    if type(min) != int and type(min) != float:
        raise TypeError("Expected 'float' or 'int' for `f`, saw '%s' instead." %
                        type(min).__name__)

    if min < 0.0 or min > 1.0:
        raise ValueError("Minimum correlation has to be a float in the range "
                         "(0.0, 1.0), not '%f'." % min)

    filter_irrelevant = lambda (i, c): i != '_Y' and abs(c) >= min
    """callable: Returns `True` if the given index belongs to a feature with a
    large enough correlation, `False` otherwise."""

    df = pd.DataFrame(X)
    """DataFrame: Pandas feature snapshot."""

    # Set observations to special key '_Y'.
    df["_Y"] = np.asarray(Y)

    try:
        indices = list(zip(*filter(filter_irrelevant,
                                   df.corr()["_Y"].iteritems()))[0])
        """list of int: Indices of relevant features."""

        X_hat = compose(np.matrix, np.zeros)((X.shape[0], len(indices)))
        """np.matrix: Reduced feature set."""

        k = 0
        """int: Feature number into reduce matrix."""

        for i in range(X.shape[1]):
            if len(indices) and i == indices[0]:
                X_hat[:, k] = X[:, indices.pop(0)]
                k += 1
            elif names is not None:
                names.pop(k)

        return X_hat
    except IndexError as e:
        raise ValueError("No features satisfy the given minimum correlation.")

def shuffle_batches(batches):
    """Batch Shuffler.

    Re-orders the datapoints in the given batch set into a new set of batches.

    Args:
        batches: Batch set to re-order.

    Returns:
        list of np.matrix: Shuffled batches.

    Raises:
        InvalidFeatureSetError: If not all the batches are compatible and valid.
        TypeError: If `batches` is not an iterable of np.matrix instances.

    """
    if hasattr(batches, 'shape'):
        raise TypeError("Expected 'list' of 'np.matrix', "
                        "saw 'np.matrix' instead.")

    extract_length = lambda b: b.shape[0]
    """callable: Returns the number of rows in the given matrix."""

    try:
        lengths = map(extract_length, batches)
        """list of (int, int): Matrix dimensions of all batches."""
    except AttributeError:
        raise TypeError("Expected iterable of np.matrix instances.")

    d_hat = batches[0].shape[1]
    """int: Number of features `d` plus 1."""

    datapoints = []
    """list of np.matrix: List representation of all data points."""
    shuffled_batches = []
    """list of np.matrix: Newly re-ordered batches."""

    for b in batches:
        for i, datapoint in enumerate(b):
            datapoints.append(datapoint)

    shuffle(datapoints)

    if d_hat < 2:
        reason = ("No features found in given dataset of shape '(%d, %d)'." %
                  batches[0].shape)
        raise InvalidFeatureSetError(batches[0], reason=reason)

    while len(lengths):
        length = lengths.pop()
        """(int, int): Current batch's matrix dimensions."""
        batch = compose(np.matrix, np.zeros)((length, d_hat))
        """int: Current batch."""

        for k in range(length):
            datapoint = datapoints.pop()

            try:
                batch[k, :] = datapoint[0, :]
            except ValueError as e:
                reason = ("No features found in given dataset of shape "
                          "'(%d, %d)'." % (batch.shape[0], datapoint.shape[1]))
                raise InvalidFeatureSetError(batch, reason=reason)

        shuffled_batches.append(batch)

    return shuffled_batches

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
        raise IncompatibleDataSetsError(X, Y, **kwargs)

def validate_feature_set(X):
    """Dataset Validator.

    Validates the given feature set for type and non-triviality.

    Args:
        X (np.matrix): Feature set. Shape: n x d.

    Raises:
        InvalidFeatureSetError: If `X` is an invalid or trivial matrix.

    """
    if type(X) != np.matrix:
        raise InvalidFeatureSetError(X, isType=True)

    if X.size == 0:
        raise InvalidFeatureSetError(X, reason="Received empty matrix.")

def validate_observation_set(Y):
    """Dataset Validator.

    Validates the given observation set for type and non-triviality.

    Args:
        Y (np.matrix): Observation set. Shape: n x 1.

    Raises:
        InvalidFeatureSetError: If `Y` is an invalid or trivial matrix or if
            `Y` does not have exactly one column.

    """
    if type(Y) != np.matrix:
        raise InvalidObservationSetError(Y, isType=True)

    if Y.size == 0:
        raise InvalidObservationSetError(Y, reason="Received empty matrix.")

    if Y.shape[1] != 1:
        raise InvalidObservationSetError(Y, reason="Not a column vector.")
