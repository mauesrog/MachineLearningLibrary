"""Statistical Utilities.

Provides the implementation for commponly used routines in ML.

"""
import numpy as _np
from random import shuffle as _shuffle
import sys as _sys
import unittest as _unittest

from test import ModuleTestCase as _ModuleTestCase
from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError, \
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
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if type(Y) != _np.matrix:
        raise _InvalidObservationSetError(Y, isType=True)

    if X.size == 0:
        raise _InvalidFeatureSetError(X, reason="Received empty matrix.")

    if Y.size == 0:
        raise _InvalidObservationSetError(Y, reason="Received empty matrix.")

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
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if X.size == 0:
        raise _InvalidFeatureSetError(X, reason="Received empty matrix.")

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
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if type(Y) != _np.matrix:
        raise _InvalidObservationSetError(Y, isType=True)

    if X.size == 0:
        raise _InvalidFeatureSetError(X, reason="Received empty matrix.")

    if Y.size == 0:
        raise _InvalidObservationSetError(Y, reason="Received empty matrix.")

    if Y.shape[1] != 1:
        reason = "Non-unit column number: %s." % (Y.shape,)
        raise _InvalidObservationSetError(Y, reason=reason)

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


class _Test(_ModuleTestCase):
    """Statistical Utilities Unit Tester.

    Runs tests for all functions exported from this module:
        - `batches`
        - `normalize`
        - `partition_data`

    Attributes:
        cutoff_zero (float): The largest value treated as zero in all equality
            tests.
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the statistical utilities module.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.

    """
    def setUp(self):
        """Statistical Utilities Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.cutoff_zero = 1e-9
        self.data_shape = (100, 20)
        self.label = '`stats`'
        self.n_tests = 20
        self.name = __name__

    def test_edge_cases_batches(self):
        """`stats.batches`: Edge Case Validator.

        Tests the behavior of `batches` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            batches(_np.matrix([[]]), _np.matrix(_np.zeros((n, 1))), 20)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix `Y`.
            batches(_np.matrix(_np.zeros((n, d))), _np.matrix([[]]), 20)

    def test_edge_cases_normalize(self):
        """`stats.normalize`: Edge Case Validator.

        Tests the behavior of `normalize` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            normalize(_np.matrix([[]]))

    def test_edge_cases_partition_data(self):
        """`stats.partition_data`: Edge Case Validator.

        Tests the behavior of `partition_data` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            partition_data(_np.matrix([[]]), _np.matrix(_np.zeros((n, 1))), 0.5)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix `Y`.
            partition_data(_np.matrix(_np.zeros((n, d))), _np.matrix([[]]), 0.5)


    def test_invalid_args_batches(self):
        """`stats.batches`: Argument Validator.

        Tests the behavior of `batches` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(TypeError):
            # No arguments.
            batches()

        with self.assertRaises(TypeError):
            # Only two arguments.
            batches(123, 123)

        with self.assertRaises(TypeError):
            # More than three arguments.
            batches(123, 123, 123, 1123)

        with self.assertRaises(_InvalidFeatureSetError):
            # List instead of matrix `X`.
            batches([[]], _np.matrix(_np.zeros((n, 1))), 20)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of matrix `Y`.
            batches(_np.matrix(_np.zeros((n, d))), None, 20)

        with self.assertRaises(TypeError):
            # String instead of positive integer `k`.
            batches(_np.matrix(_np.zeros((n, d))),
                    _np.matrix(_np.zeros((n, 1))), "hello")

        with self.assertRaises(TypeError):
            # List of integers instead of positive integer `k`.
            batches(_np.matrix(_np.zeros((n, d))),
                    _np.matrix(_np.zeros((n, 1))), [0])

        with self.assertRaises(TypeError):
            # Matrix instead of positive integer `k`.
            batches(_np.matrix(_np.zeros((n, d))),
                    _np.matrix(_np.zeros((n, 1))),
                    _np.matrix(_np.zeros((n, 1))))

        with self.assertRaises(TypeError):
            # Float instead of positive integer `k`.
            batches(_np.matrix(_np.zeros((n, d))),
                    _np.matrix(_np.zeros((n, 1))), 10.5)

        with self.assertRaises(ValueError):
            # Zero-valued `k`.
            batches(_np.matrix(_np.random.rand(n, d)),
                    _np.matrix(_np.random.rand(n, 1)), 0)

        with self.assertRaises(ValueError):
            # Negative `k`.
            batches(_np.matrix(_np.random.rand(n, d)),
                    _np.matrix(_np.random.rand(n, 1)), -1)

    def test_invalid_args_normalize(self):
        """`stats.normalize`: Argument Validator.

        Tests the behavior of `normalize` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(TypeError):
            # No arguments.
            normalize()

        with self.assertRaises(TypeError):
            # More than one argument.
            normalize(123, 123, 123)

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            normalize(None)

        with self.assertRaises(_InvalidFeatureSetError):
            # List of an empty list instead of matrix `X`.
            normalize([[]])

    def test_invalid_args_partition_data(self):
        """`stats.partition_data`: Argument Validator.

        Tests the behavior of `partition_data` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(TypeError):
            # Only two arguments.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, d))))

        with self.assertRaises(TypeError):
            # Only one arguments.
            partition_data(0)

        with self.assertRaises(TypeError):
            # No arguments.
            partition_data()

        with self.assertRaises(TypeError):
            # More than three arguments.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, d))), 0.5, 123)

        with self.assertRaises(_InvalidFeatureSetError):
            # List of a matrix instead of matrix `X`.
            partition_data([_np.matrix(_np.zeros((n, d)))],
                           _np.matrix(_np.zeros((n, 1))), 0.2)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of matrix `Y`.
            partition_data(_np.matrix(_np.zeros((n, 1))), None, 0.5)

        with self.assertRaises(_InvalidObservationSetError):
            # Incompatible observation matrix `Y`.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, d))), 0.5)

        with self.assertRaises(_InvalidFeatureSetError):
            # Lists instead of matrices `X` and `Y`.
            partition_data([], [], 0.5)

        with self.assertRaises(TypeError):
            # None instead of float `f`.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, 1))), None)

        with self.assertRaises(ValueError):
            # Ratio `f` greater than 1.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, 1))), 1.2)

        with self.assertRaises(ValueError):
            # Zero-valued ratio `f`.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, 1))), 0)

        with self.assertRaises(ValueError):
            # Unit-valued ratio `f`.
            partition_data(_np.matrix(_np.zeros((n, d))),
                           _np.matrix(_np.zeros((n, 1))), 1)

    def test_random_batches(self):
        """`stats.batches`: Randomized Validator.

        Tests the behavior of `batches` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `batches` needs debugging.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""
        step_size = int(_np.floor(n / float(self.n_tests)))
        """int: Amount by which to increase batch-size `k` after each
        iteration."""

        for k in range(5, n + 10, step_size):
            X = _np.matrix(_np.random.rand(n, d))
            """np.matrix: Random-valued feature set."""
            Y = _np.matrix(_np.random.rand(n, 1))
            """np.matrix: Random-valued observation set."""

            buckets = batches(X, Y, k)
            """list of np.matrix: Test input."""

            # `buckets` should be a list.
            self.assertEqual(type(buckets), list)

            # Each bucket `b` in `buckets` should be a matrix.
            for b in buckets:
                self.assertEqual(type(b), _np.matrix)

            # Number of buckets should match the total number of data points
            # (floor) divided by the number of data points per bucket.
            self.assertEqual(len(buckets), int(_np.floor(n / float(min(n, k)))))

            # Total number of data points across all buckets should match
            # original number of data points.
            self.assertEqual(sum([b.shape[0] for b in buckets]), n)

            norm_buckets = sum([sum([_np.linalg.norm(b[j, :]) ** 2
                                     for j in range(0, b.shape[0])])
                                for b in buckets])
            """float: Sum of the sum of the norms of all rows in all buckets."""
            norm = sum([_np.linalg.norm(X[i, :]) ** 2 + Y[i, 0] ** 2
                        for i in range(0, n)])
            """float: Sum of the norms of all rows in feature set plus the
            square of all data points in observation set."""

            # The norms of all bucket rows and the norms of the original dataset
            # should match when summed.
            self.assertLessEqual(abs(norm_buckets - norm), self.cutoff_zero)

    def test_random_normalize(self):
        """`stats.normalize`: Randomized Validator.

        Tests the behavior of `normalize` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `normalize` needs debugging.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        for i in range(0, self.n_tests):
            X = normalize(_np.matrix(_np.random.rand(n, d)))
            """np.matrix: Randomized test input."""

            # `X` should be a matrix.
            self.assertEqual(type(X), _np.matrix)

            # The means of all rows in `X` should be zero-valued.
            self.assertLessEqual(_np.linalg.norm(_np.mean(X, 1)) ** 2,
                                 self.cutoff_zero)

            # The standard deviations of all rows in `X` should be unit-valued.
            self.assertEqual(_np.linalg.norm(_np.std(X, 1)) ** 2, n)

    def test_random_partition_data(self):
        """`stats.partition_data`: Randomized Validator.

        Tests the behavior of `partition_data` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `partition_data` needs debugging.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""
        step_size = int(_np.floor(n / float(self.n_tests)))
        """int: Amount by which to increase percentage `f` after each
        iteration."""

        for f in range(5, n, step_size):
            X = _np.matrix(_np.random.rand(n, d))
            """np.matrix: Random-valued feature set."""
            Y = _np.matrix(_np.random.rand(n, 1))
            """np.matrix: Random-valued observation set."""

            # Transform percentage into decimal ratio.
            f *= 1e-2

            train_X, train_Y, test_X, test_Y = partition_data(X, Y, f)
            """(np.matrix, np.matrix, np.matrix, np.matrix): Test input."""

            # All partitions should be matrices.
            self.assertEqual(type(train_X), _np.matrix)
            self.assertEqual(type(train_Y), _np.matrix)
            self.assertEqual(type(test_X), _np.matrix)
            self.assertEqual(type(test_Y), _np.matrix)

            # Number of training data points should match the total number of
            # data points (floor) multiplied by ratio `f`.
            self.assertEqual(train_X.shape[0], int(_np.floor(n * f)))
            self.assertEqual(train_Y.shape[0], int(_np.floor(n * f)))

            # The number of data points across all parititons should match that
            # of the original whole.
            self.assertEqual(test_X.shape[0] + train_X.shape[0], n)
            self.assertEqual(test_Y.shape[0] + train_Y.shape[0], n)

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: Statistical utilities testing suite."""
