"""Statistical Utilities Testing Module.

Attributes:
    Test (TestSuite): Statistical utilities testing suite.

"""
import numpy as _np
import unittest as _unittest

from config import data_examples as _examples
from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidObservationSetError as _InvalidObservationSetError
from stats import batches, normalize, partition_data, reduce_dimensions, \
                  shuffle_batches
from linalg import random_matrix as _random_matrix
from general import appendargs as _appendargs, compose as _compose, \
                    prependargs as _prependargs


class _Test(_ModuleTestCase):
    """Statistical Utilities Unit Tester.

    Runs tests for all functions exported from this module:
        - `batches`
        - `normalize`
        - `partition_data`

    Attributes:
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
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            batches(_np.matrix([[]]), Y, 20)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix `Y`.
            batches(X, _np.matrix([[]]), 20)

        with self.assertRaises(ValueError):
            # `k` greater than 'n'.
            print len(batches(X, Y, 110))

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
            partition_data(_np.matrix([[]]), _random_matrix((n, 1)), 0.5)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix `Y`.
            partition_data(_random_matrix((n, d)), _np.matrix([[]]), 0.5)

    def test_edge_cases_reduce_dimensions(self):
        """`stats.reduce_dimensions`: Edge Case Validator.

        Tests the behavior of `reduce_dimensions` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            reduce_dimensions(_np.matrix([[]]), _random_matrix((n, 1)))

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix `Y`.
            reduce_dimensions(_random_matrix((n, d)), _np.matrix([[]]))

    def test_edge_cases_shuffle_batches(self):
        """`stats.shuffle_batches`: Edge Case Validator.

        Tests the behavior of `shuffle_batches` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Batches all empty.
            shuffle_batches([_np.matrix([[]]), _np.matrix([[]])])

        with self.assertRaises(_InvalidFeatureSetError):
            # Second batch empty.
            shuffle_batches([X, _np.matrix([[]])])

        with self.assertRaises(_InvalidFeatureSetError):
            # Third batch empty.
            shuffle_batches([X, X, _np.matrix([[]])])

    def test_example_reduce_dimensions(self):
        """`stats.reduce_dimensions`: Example Validator.

        Tests the behavior of `reduce_dimensions` by feeding it real-life
        datasets.

        Raises:
            AssertionError: If `train` needs debugging.

        """
        for types in _examples.values():
            for type, loader in types.iteritems():
                if type == "classification":
                    break

                X, Y = loader()[:2]
                """(np.matrix, np.matrix): Real datasets."""
                X_hat = reduce_dimensions(X, Y, 0.5)
                """np.matrix: Feature set with reduced features."""

                # Number of data points should have remained unchanged.
                self.assertEqual(X.shape[0], X_hat.shape[0])

                # There should be less features per data point now.
                self.assertLess(X_hat.shape[1], X.shape[1])


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
            batches([[]], _random_matrix((n, 1)), 20)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of matrix `Y`.
            batches(_random_matrix((n, d)), None, 20)

        with self.assertRaises(TypeError):
            # String instead of positive integer `k`.
            batches(_random_matrix((n, d)),
                    _random_matrix((n, 1)), "hello")

        with self.assertRaises(TypeError):
            # List of integers instead of positive integer `k`.
            batches(_random_matrix((n, d)),
                    _random_matrix((n, 1)), [0])

        with self.assertRaises(TypeError):
            # Matrix instead of positive integer `k`.
            batches(_random_matrix((n, d)),
                    _random_matrix((n, 1)),
                    _random_matrix((n, 1)))

        with self.assertRaises(TypeError):
            # Float instead of positive integer `k`.
            batches(_random_matrix((n, d)),
                    _random_matrix((n, 1)), 10.5)

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
            partition_data(_random_matrix((n, d)), _random_matrix((n, d)))

        with self.assertRaises(TypeError):
            # Only one arguments.
            partition_data(0)

        with self.assertRaises(TypeError):
            # No arguments.
            partition_data()

        with self.assertRaises(TypeError):
            # More than three arguments.
            partition_data(_random_matrix((n, d)), _random_matrix((n, d)), 0.5, 123)

        with self.assertRaises(_InvalidFeatureSetError):
            # List of a matrix instead of matrix `X`.
            partition_data([_random_matrix((n, d))], _random_matrix((n, 1)), 0.2)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of matrix `Y`.
            partition_data(_random_matrix((n, 1)), None, 0.5)

        with self.assertRaises(_InvalidObservationSetError):
            # Incompatible observation matrix `Y`.
            partition_data(_random_matrix((n, d)), _random_matrix((n, d)), 0.5)

        with self.assertRaises(_InvalidFeatureSetError):
            # Lists instead of matrices `X` and `Y`.
            partition_data([], [], 0.5)

        with self.assertRaises(TypeError):
            # None instead of float `f`.
            partition_data(_random_matrix((n, d)), _random_matrix((n, 1)), None)

        with self.assertRaises(ValueError):
            # Ratio `f` greater than 1.
            partition_data(_random_matrix((n, d)), _random_matrix((n, 1)), 1.2)

        with self.assertRaises(ValueError):
            # Zero-valued ratio `f`.
            partition_data(_random_matrix((n, d)), _random_matrix((n, 1)), 0)

        with self.assertRaises(ValueError):
            # Unit-valued ratio `f`.
            partition_data(_random_matrix((n, d)), _random_matrix((n, 1)), 1)

    def test_invalid_args_reduce_dimensions(self):
        """`stats.reduce_dimensions`: Argument Validator.

        Tests the behavior of `reduce_dimensions` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        with self.assertRaises(TypeError):
            # No arguments.
            reduce_dimensions()

        with self.assertRaises(TypeError):
            # Only one argument.
            reduce_dimensions(X)

        with self.assertRaises(TypeError):
            # More than three arguments.
            reduce_dimensions(X, Y, 0.5, None, "extra_arg")

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            reduce_dimensions(X, Y, 0.5, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            reduce_dimensions(_np.zeros(self.data_shape), Y)

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            reduce_dimensions(None, Y)

        with self.assertRaises(_InvalidObservationSetError):
            # ndarray instead of matrix `Y`.
            reduce_dimensions(X, _np.zeros((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of matrix `Y`.
            reduce_dimensions(X, None)

        with self.assertRaises(_InvalidObservationSetError):
            # Incompatible data sets.
            reduce_dimensions(X, X)

        with self.assertRaises(TypeError):
            # None instead of float `f`.
            reduce_dimensions(X, Y, None)

        with self.assertRaises(ValueError):
            # Minimum correlation greater than 1.
            reduce_dimensions(X, Y, 1.2)

        with self.assertRaises(ValueError):
            # Negative minimum correlation.
            reduce_dimensions(X, Y, -0.1)

    def test_invalid_args_shuffle_batches(self):
        """`stats.shuffle_batches`: Argument Validator.

        Tests the behavior of `shuffle_batches` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        b = batches(X, Y, 25)
        """list of np.matrix: Random-valued batches."""

        with self.assertRaises(TypeError):
            # No arguments.
            shuffle_batches()

        with self.assertRaises(TypeError):
            # Too many arguments.
            shuffle_batches(b, "extra")

        with self.assertRaises(TypeError):
            # Keyword argument.
            shuffle_batches(b, key="value")

        with self.assertRaises(TypeError):
            # Matrix instead of list `batches`.
            shuffle_batches(b[0])

        with self.assertRaises(TypeError):
            # Non-iterable `batches`.
            shuffle_batches(None)

        with self.assertRaises(TypeError):
            # Wrong kind of iterable `batches`.
            shuffle_batches("string")

    def test_random_batches(self):
        """`stats.batches`: Randomized Validator.

        Tests the behavior of `batches` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `batches` needs debugging.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""
        step_size = _compose(int, _np.floor)((n - 1) / float(self.n_tests))
        """int: Amount by which to increase batch-size `k` after each
        iteration."""

        for k in range(1, n, step_size):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            Y = _random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            target_len = max(2, int(_np.floor(n / float(min(n, k)))))
            """int: Expected number of batches."""

            buckets = batches(X, Y, k)
            """list of np.matrix: Test input."""

            # `buckets` should be a list.
            self.assertIsInstance(buckets, list)

            # Each bucket `b` in `buckets` should be a matrix.
            for b in buckets:
                self.assertIsInstance(b, _np.matrix)

            # Number of buckets should match the total number of data points
            # (floor) divided by the number of data points per bucket.
            self.assertEqual(len(buckets), target_len)

            # Total number of data points across all buckets should match
            # original number of data points.
            self.assertEqual(sum([b.shape[0] for b in buckets]), n)

            norm_buckets = sum([sum([_np.linalg.norm(b[j, :]) ** 2
                                     for j in range(b.shape[0])])
                                for b in buckets])
            """float: Sum of the sum of the norms of all rows in all buckets."""
            norm = sum([_np.linalg.norm(X[i, :]) ** 2 + Y[i, 0] ** 2
                        for i in range(n)])
            """float: Sum of the norms of all rows in feature set plus the
            square of all data points in observation set."""

            # The norms of all bucket rows and the norms of the original dataset
            # should match when summed.
            self.assertAlmostEqual(norm_buckets, norm)

    def test_random_normalize(self):
        """`stats.normalize`: Randomized Validator.

        Tests the behavior of `normalize` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `normalize` needs debugging.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        for i in range(self.n_tests):
            X = normalize(_np.matrix(_np.random.rand(n, d)))
            """np.matrix: Randomized test input."""

            # `X` should be a matrix.
            self.assertIsInstance(X, _np.matrix)

            # The means of all rows in `X` should be zero-valued.
            self.assertLessEqual(_np.linalg.norm(_np.mean(X, 1)) ** 2,
                                 self.zero_cutoff)

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
            self.assertIsInstance(train_X, _np.matrix)
            self.assertIsInstance(train_Y, _np.matrix)
            self.assertIsInstance(test_X, _np.matrix)
            self.assertIsInstance(test_Y, _np.matrix)

            # Number of training data points should match the total number of
            # data points (floor) multiplied by ratio `f`.
            self.assertEqual(train_X.shape[0], int(_np.floor(n * f)))
            self.assertEqual(train_Y.shape[0], int(_np.floor(n * f)))

            # The number of data points across all parititons should match that
            # of the original whole.
            self.assertEqual(test_X.shape[0] + train_X.shape[0], n)
            self.assertEqual(test_Y.shape[0] + train_Y.shape[0], n)

    def test_random_shuffle_batches(self):
        """`stats.shuffle_batches`: Randomized Validator.

        Tests the behavior of `shuffle_batches` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `shuffle_batches` needs debugging.

        """
        extract_shape = lambda b: b.shape
        """callable: Returns the dimensions of the given matrix."""
        row_norms = lambda b: _np.linalg.norm(b, axis=1)
        """callable: Returns the sum of all row norms in `b`."""
        aslist = lambda A: A.T[0, :].tolist()[0]
        """callable: Maps vectors to lists."""
        is_row_equal = lambda r: _compose(all, aslist)(r[0, :])
        """callable: Determines whether all entries are set to `True` in given
        element-wise equality row vector."""
        are_rows_equal = lambda v: v if isinstance(v, bool) \
                                     else _prependargs(_compose(all, map),
                                                       is_row_equal)(v)
        """callable: Determines whether all entries are set to `True` in given
        element-wise equality matrix vector."""

        for i in range(self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            Y = _random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            b = batches(X, Y, 29)
            """list of np.matrix: Random-valued batches."""
            b_hat = shuffle_batches(b)
            """list of np.matrix: Test input."""

            shapes = map(extract_shape, b)
            """list of (int, int): Input batch shapes."""
            shapes_hat = map(extract_shape, b_hat)
            """list of (int, int): Output batch shapes."""



            # Re-ordered batches should be a list of np.matrix instances.
            self.assertIsInstance(b_hat, list)
            map(_appendargs(self.assertIsInstance, _np.matrix), b_hat)

            # Shapes and norms should match.
            self.assertTrue(_compose(all,
                                     map)(lambda shape: shape in shapes,
                                          shapes_hat))
            self.assertAlmostEqual(*map(_prependargs(_compose(sum, map),
                                                     _compose(sum, row_norms)),
                                        [b, b_hat]))

            # Batches should be in different order.
            batches_not_equal = map(lambda (b1, b2): b1 != b2, zip(b, b_hat))
            """np.matrix: Element-wise equality matrix of all batches."""
            _compose(self.assertTrue, any, map)(are_rows_equal,
                                                batches_not_equal)

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
