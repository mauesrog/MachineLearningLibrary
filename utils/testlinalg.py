"""Linear Algebra Utilities Testing Module.

"""
import math as _math
import numpy as _np
import unittest as _unittest

from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError
from general import compose
from linalg import append_bottom, append_left, append_right, append_top, \
                   diagonal, random_matrix, _append_helper


class _Test(_ModuleTestCase):
    """Linear Algebra Utilities Unit Tester.

    Runs tests for all linear albegra utilties:
        - `diagonal`

    Attributes:
        cutoff_zero (float): The largest value treated as zero in all equality
            tests.
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the general model utilities module. See base
            class.
        max_val (int): Largest random number allowed.
        n (int): Number of data points to use in any test e.g. in random tests.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.

    """
    def setUp(self):
        """Linear Algebra Utilities Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.cutoff_zero = 1e-6
        self.label = '`linalg`'
        self.max_val = 100
        self.data_shape = 100, 20
        self.n_tests = 20
        self.name = __name__

    def test_edge_cases_append_bottom(self):
        """`linalg.append_bottom`: Edge Case Validator.

        Tests the behavior of `append_bottom` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        # Empty matrix `X`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_bottom(_np.matrix([[]]), random_matrix(self.data_shape))

        # Empty vector `v`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_bottom(random_matrix(self.data_shape), _np.matrix([[]]))

    def test_edge_cases_append_left(self):
        """`linalg.append_left`: Edge Case Validator.

        Tests the behavior of `append_left` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        # Empty matrix `X`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_left(_np.matrix([[]]), random_matrix(self.data_shape))

        # Empty vector `v`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_left(random_matrix(self.data_shape), _np.matrix([[]]))

    def test_edge_cases_append_right(self):
        """`linalg.append_right`: Edge Case Validator.

        Tests the behavior of `append_right` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        # Empty matrix `X`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_right(_np.matrix([[]]), random_matrix(self.data_shape))

        # Empty vector `v`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_right(random_matrix(self.data_shape), _np.matrix([[]]))

    def test_edge_cases_append_top(self):
        """`linalg.append_top`: Edge Case Validator.

        Tests the behavior of `append_top` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        # Empty matrix `X`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_top(_np.matrix([[]]), random_matrix(self.data_shape))

        # Empty vector `v`.
        with self.assertRaises(_InvalidFeatureSetError):
            append_top(random_matrix(self.data_shape), _np.matrix([[]]))

    def test_edge_cases_diagonal(self):
        """`linalg.diagonal`: Edge Case Validator.

        Tests the behavior of `diagonal` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

    def test_edge_cases_random_matrix(self):
        """`linalg.random_matrix`: Edge Case Validator.

        Tests the behavior of `random_matrix` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

    def test_invalid_args_append_helper(self):
        """`linalg.append_helper`: Argument Validator.

        Tests the behavior of `append_helper` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(ValueError):
            # Invalid position argument.
            _append_helper(random_matrix(self.data_shape),
                           random_matrix((self.data_shape[0], 1)),
                           position="corner")

    def test_invalid_args_append_bottom(self):
        """`linalg.append_bottom`: Argument Validator.

        Tests the behavior of `append_bottom` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            append_bottom()

        with self.assertRaises(TypeError):
            # Only one argument given.
            append_bottom(random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # More arguments than needed.
            append_bottom(random_matrix(self.data_shape),
                          random_matrix((1, self.data_shape[1])),
                          random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # With **kwargs.
            append_bottom(random_matrix(self.data_shape),
                          random_matrix((1, self.data_shape[1])), key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of matrix `X`.
            append_bottom("string", random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            append_bottom(None, random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            append_bottom(_np.random.uniform(size=self.data_shape[1]),
                          random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of vector `v`.
            append_bottom(random_matrix(self.data_shape), "string")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of vector `v`.
            append_bottom(random_matrix(self.data_shape), None)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of vector `v`.
            append_bottom(random_matrix(self.data_shape),
                          _np.random.uniform(size=(1, self.data_shape[1])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_bottom(random_matrix(self.data_shape),
                          random_matrix((self.data_shape[1], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_bottom(random_matrix((self.data_shape[1],
                                         self.data_shape[0])),
                          random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Matrix instead of vector `v`.
            append_bottom(random_matrix(self.data_shape),
                          random_matrix(self.data_shape))

    def test_invalid_args_append_left(self):
        """`linalg.append_left`: Argument Validator.

        Tests the behavior of `append_left` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            append_left()

        with self.assertRaises(TypeError):
            # Only one argument given.
            append_left(random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # More arguments than needed.
            append_left(random_matrix(self.data_shape),
                        random_matrix((self.data_shape[0], 1)),
                        random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # With **kwargs.
            append_left(random_matrix(self.data_shape),
                        random_matrix((self.data_shape[0], 1)), key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of matrix `X`.
            append_left("string", random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            append_left(None, random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            append_left(_np.random.uniform(size=self.data_shape[0]),
                        random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of vector `v`.
            append_left(random_matrix(self.data_shape), "string")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of vector `v`.
            append_left(random_matrix(self.data_shape), None)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of vector `v`.
            append_left(random_matrix(self.data_shape),
                        _np.random.uniform(size=(self.data_shape[0], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_left(random_matrix(self.data_shape),
                        random_matrix((1, self.data_shape[0])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_left(random_matrix((self.data_shape[1], self.data_shape[0])),
                        random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Matrix instead of vector `v`.
            append_left(random_matrix(self.data_shape),
                        random_matrix(self.data_shape))

    def test_invalid_args_append_right(self):
        """`linalg.append_right`: Argument Validator.

        Tests the behavior of `append_right` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            append_right()

        with self.assertRaises(TypeError):
            # Only one argument given.
            append_right(random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # More arguments than needed.
            append_right(random_matrix(self.data_shape),
                         random_matrix((self.data_shape[0], 1)),
                         random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # With **kwargs.
            append_right(random_matrix(self.data_shape),
                         random_matrix((self.data_shape[0], 1)), key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of matrix `X`.
            append_right("string", random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            append_right(None, random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            append_right(_np.random.uniform(size=self.data_shape[0]),
                         random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of vector `v`.
            append_right(random_matrix(self.data_shape), "string")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of vector `v`.
            append_right(random_matrix(self.data_shape), None)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of vector `v`.
            append_right(random_matrix(self.data_shape),
                         _np.random.uniform(size=(self.data_shape[0], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_right(random_matrix(self.data_shape),
                         random_matrix((1, self.data_shape[0])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_right(random_matrix((self.data_shape[1],
                                        self.data_shape[0])),
                         random_matrix((self.data_shape[0], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Matrix instead of vector `v`.
            append_right(random_matrix(self.data_shape),
                         random_matrix(self.data_shape))

    def test_invalid_args_append_top(self):
        """`linalg.append_top`: Argument Validator.

        Tests the behavior of `append_top` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            append_top()

        with self.assertRaises(TypeError):
            # Only one argument given.
            append_top(random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # More arguments than needed.
            append_top(random_matrix(self.data_shape),
                       random_matrix((1, self.data_shape[1])),
                       random_matrix(self.data_shape))

        with self.assertRaises(TypeError):
            # With **kwargs.
            append_top(random_matrix(self.data_shape),
                       random_matrix((1, self.data_shape[1])), key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of matrix `X`.
            append_top("string", random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            append_top(None, random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            append_top(_np.random.uniform(size=self.data_shape[1]),
                       random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of vector `v`.
            append_top(random_matrix(self.data_shape), "string")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of vector `v`.
            append_top(random_matrix(self.data_shape), None)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of vector `v`.
            append_top(random_matrix(self.data_shape),
                       _np.random.uniform(size=(1, self.data_shape[1])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_top(random_matrix(self.data_shape),
                       random_matrix((self.data_shape[1], 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible matrices `X` and `v`.
            append_top(random_matrix((self.data_shape[1], self.data_shape[0])),
                       random_matrix((1, self.data_shape[1])))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Matrix instead of vector `v`.
            append_top(random_matrix(self.data_shape),
                       random_matrix(self.data_shape))

    def test_invalid_args_diagonal(self):
        """`linalg.diagonal`: Argument Validator.

        Tests the behavior of `diagonal` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            diagonal()

        with self.assertRaises(TypeError):
            # Only optional argument given.
            diagonal(val=_np.random.uniform())

        with self.assertRaises(TypeError):
            # More arguments than needed.
            diagonal(self.data_shape[0], _np.random.uniform(), "extra")

        with self.assertRaises(TypeError):
            # Invalid *kwarg.
            diagonal(self.data_shape[0], _np.random.uniform(), key="val")

        with self.assertRaises(TypeError):
            # String instead of integer `n`.
            diagonal("n")

        with self.assertRaises(TypeError):
            # `None` instead of integer `n`.
            diagonal(None)

        with self.assertRaises(TypeError):
            # Float instead of integer `n`.
            diagonal(1e-6)

        with self.assertRaises(TypeError):
            # Integer tuple instead of integer `n`.
            diagonal((self.data_shape[0],))

        with self.assertRaises(ValueError):
            # Zero-valued integer `n`.
            diagonal(0)

        with self.assertRaises(ValueError):
            # Negative integer `n`.
            diagonal(-14)

        with self.assertRaises(TypeError):
            # String instead of optional float `val`.
            diagonal(self.data_shape[0], "val")

        with self.assertRaises(TypeError):
            # `None` instead of optional float `val`.
            diagonal(self.data_shape[0], None)

        with self.assertRaises(TypeError):
            # Tuple instead of optional float `val`.
            diagonal(self.data_shape[0], val=(0.2123,))

    def test_invalid_args_random_matrix(self):
        """`linalg.random_matrix`: Argument Validator.

        Tests the behavior of `random_matrix` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            random_matrix()

        with self.assertRaises(TypeError):
            # Only optional argument given.
            random_matrix(min_val=0.2)

        with self.assertRaises(TypeError):
            # More arguments than needed.
            random_matrix(self.data_shape, 1.0, 100.0, "extra")

        with self.assertRaises(TypeError):
            # Invalid *kwarg.
            random_matrix(self.data_shape, key="val")

        with self.assertRaises(TypeError):
            # `None` instead of integer tuple `size`.
            random_matrix(None)

        with self.assertRaises(TypeError):
            # Integer instead of integer tuple `size`.
            random_matrix(1)

        with self.assertRaises(TypeError):
            # Float instead of integer tuple `size`.
            random_matrix(1e-6)

        with self.assertRaises(ValueError):
            # Float tuple instead of integer tuple `size`.
            random_matrix((100.5, 5.34))

        with self.assertRaises(ValueError):
            # Empty integer tuple.
            random_matrix(())

        with self.assertRaises(ValueError):
            # Insufficient integer tuple length.
            random_matrix((100,))

        with self.assertRaises(ValueError):
            # Exceeding integer tuple length.
            random_matrix((100, 5, 123))

        with self.assertRaises(ValueError):
            # Zero-valued row number.
            random_matrix((0, self.data_shape[1]))

        with self.assertRaises(ValueError):
            # Negative row number.
            random_matrix((-12, self.data_shape[1]))

        with self.assertRaises(ValueError):
            # Zero-valued column number.
            random_matrix((self.data_shape[0], 0))

        with self.assertRaises(ValueError):
            # Negative column number.
            random_matrix((self.data_shape[0], -12))

        with self.assertRaises(TypeError):
            # String instead of optional float `min_val`.
            random_matrix(self.data_shape, "val")

        with self.assertRaises(TypeError):
            # `None` instead of optional float `min_val`.
            random_matrix(self.data_shape, None)

        with self.assertRaises(TypeError):
            # Float tuple instead of optional float `min_val`.
            random_matrix(self.data_shape, (0.2123,))

        with self.assertRaises(TypeError):
            # String instead of optional float `max_val`.
            random_matrix(self.data_shape, max_val="val")

        with self.assertRaises(TypeError):
            # `None` instead of optional float `max_val`.
            random_matrix(self.data_shape, max_val=None)

        with self.assertRaises(TypeError):
            # Float tuple instead of optional float `max_val`.
            random_matrix(self.data_shape, max_val=(0.2123,))

        with self.assertRaises(ValueError):
            # Incompatible bounds.
            random_matrix(self.data_shape, min_val=10, max_val=5)

    def test_random_diagonal(self):
        """`linalg.diagonal`: Randomized Validator.

        Tests the behavior of `diagonal` by feeding it randomly generated arguments.

        Raises:
            AssertionError: If `diagonal` needs debugging.

        """
        for i in range(0, self.n_tests):
            n = _np.random.randint(1, self.max_val)
            """int: Random-valued diagonal vector length."""
            val = _np.random.uniform(0.0, float(self.max_val))
            """float: Random-valued non-zero value."""

            result = diagonal(n, val)
            """np.matrix: Test input."""

            # Result should be a matrix.
            self.assertEqual(type(result), _np.matrix)

            # The norm of the diagonal should be computable accordin to the
            # following formula.
            self.assertLessEqual(abs(_np.linalg.norm(result.diagonal()) -
                                                     _math.sqrt(n) * val),
                                 self.cutoff_zero)

    def test_random_append_left(self):
        """`linalg.append_left`: Randomized Validator.

        Tests the behavior of `append_left` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `append_left` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            v = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued row vector."""

            result = append_left(X, v)
            """np.matrix: Test input."""

            # Result should be a matrix.
            self.assertEqual(type(result), _np.matrix)

            to_norm = lambda (A, axis): _np.linalg.norm(A, axis=axis)
            """callable: Takes in a matrix returns the norm along the specified
            axis."""

            norm_normalizer = lambda n: [n] if type(n) == _np.float64 else n
            """callable: Make sure that all norms are lists. In particular,
            treats the row vector norm as a single row of a regular matrix."""

            norms = map(to_norm, [(X, 0), (v, None), (result, 0)])
            """list: Contains the row norms of both the input and the
            augmented result."""

            # Change the sign of the augmented matrix's norm to compute norm
            # deltas and infer errors from there.
            norms[2] *= -1.0

            delta = compose(abs, sum, map)(sum, map(norm_normalizer, norms))
            """float: Absolute difference between row norms of input and those
            of the augmented matrix."""

            # The row norms of input should match those of the augmented matrix.
            self.assertLessEqual(delta, self.cutoff_zero)

            # The vector norm of `v` should match that of the leftmost row
            # vector in the augmented matrix.
            self.assertLessEqual(abs(_np.linalg.norm(v) -
                                     _np.linalg.norm(result[:, 0])),
                                 self.cutoff_zero)

    def test_random_append_right(self):
        """`linalg.append_right`: Randomized Validator.

        Tests the behavior of `append_right` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `append_right` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            v = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued row vector."""

            result = append_right(X, v)
            """np.matrix: Test input."""

            # Result should be a matrix.
            self.assertEqual(type(result), _np.matrix)

            to_norm = lambda (A, axis): _np.linalg.norm(A, axis=axis)
            """callable: Takes in a matrix returns the norm along the specified
            axis."""

            norm_normalizer = lambda n: [n] if type(n) == _np.float64 else n
            """callable: Make sure that all norms are lists. In particular,
            treats the row vector norm as a single row of a regular matrix."""

            norms = map(to_norm, [(X, 0), (v, None), (result, 0)])
            """list: Contains the row norms of both the input and the
            augmented result."""

            # Change the sign of the augmented matrix's norm to compute norm
            # deltas and infer errors from there.
            norms[2] *= -1.0

            delta = compose(abs, sum, map)(sum, map(norm_normalizer, norms))
            """float: Absolute difference between row norms of input and those
            of the augmented matrix."""

            # The row norms of input should match those of the augmented matrix.
            self.assertLessEqual(delta, self.cutoff_zero)

            # The vector norm of `v` should match that of the rightmost row
            # vector in the augmented matrix.
            self.assertLessEqual(abs(_np.linalg.norm(v) -
                                     _np.linalg.norm(result[:, -1])),
                                 self.cutoff_zero)

    def test_random_append_top(self):
        """`linalg.append_top`: Randomized Validator.

        Tests the behavior of `append_top` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `append_top` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            v = random_matrix((1, self.data_shape[1]))
            """np.matrix: Random-valued row vector."""

            result = append_top(X, v)
            """np.matrix: Test input."""

            # Result should be a matrix.
            self.assertEqual(type(result), _np.matrix)

            to_norm = lambda (A, axis): _np.linalg.norm(A, axis=axis)
            """callable: Takes in a matrix returns the norm along the specified
            axis."""

            norm_normalizer = lambda n: [n] if type(n) == _np.float64 else n
            """callable: Make sure that all norms are lists. In particular,
            treats the row vector norm as a single row of a regular matrix."""

            norms = map(to_norm, [(X, 1), (v, None), (result, 1)])
            """list: Contains the row norms of both the input and the
            augmented result."""

            # Change the sign of the augmented matrix's norm to compute norm
            # deltas and infer errors from there.
            norms[2] *= -1.0

            delta = compose(abs, sum, map)(sum, map(norm_normalizer, norms))
            """float: Absolute difference between row norms of input and those
            of the augmented matrix."""

            # The row norms of input should match those of the augmented matrix.
            self.assertLessEqual(delta, self.cutoff_zero)

            # The vector norm of `v` should match that of the topmost row
            # vector in the augmented matrix.
            self.assertLessEqual(abs(_np.linalg.norm(v) -
                                     _np.linalg.norm(result[0, :])),
                                 self.cutoff_zero)

    def test_random_append_bottom(self):
        """`linalg.append_bottom`: Randomized Validator.

        Tests the behavior of `append_bottom` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `append_bottom` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            v = random_matrix((1, self.data_shape[1]))
            """np.matrix: Random-valued row vector."""

            result = append_bottom(X, v)
            """np.matrix: Test input."""

            # Result should be a matrix.
            self.assertEqual(type(result), _np.matrix)

            to_norm = lambda (A, axis): _np.linalg.norm(A, axis=axis)
            """callable: Takes in a matrix returns the norm along the specified
            axis."""

            norm_normalizer = lambda n: [n] if type(n) == _np.float64 else n
            """callable: Make sure that all norms are lists. In particular,
            treats the row vector norm as a single row of a regular matrix."""

            norms = map(to_norm, [(X, 1), (v, None), (result, 1)])
            """list: Contains the row norms of both the input and the
            augmented result."""

            # Change the sign of the augmented matrix's norm to compute norm
            # deltas and infer errors from there.
            norms[2] *= -1.0

            delta = compose(abs, sum, map)(sum, map(norm_normalizer, norms))
            """float: Absolute difference between row norms of input and those
            of the augmented matrix."""

            # The row norms of input should match those of the augmented matrix.
            self.assertLessEqual(delta, self.cutoff_zero)

            # The vector norm of `v` should match that of the bottommost row
            # vector in the augmented matrix.
            self.assertLessEqual(abs(_np.linalg.norm(v) -
                                     _np.linalg.norm(result[-1, :])),
                                 self.cutoff_zero)

    def test_random_random_matrix(self):
        """`linalg.random_matrix`: Randomized Validator.

        Tests the behavior of `random_matrix` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `random_matrix` needs debugging.

        """
        prev = None
        """float: Holds the norm of the random matrix of the previous
        iteration."""

        for i in range(0, self.n_tests):
            min_val = _np.random.uniform(0.0, float(self.max_val))
            """float: Lower bound for random matrix."""
            max_val = _np.random.uniform(min_val + 1.0, float(self.max_val))
            """float: Upper bound for random matrix."""

            result = random_matrix(self.data_shape, min_val, max_val)
            """np.matrix: Test input."""

            # Result should be a matrix of the specified dimensions.
            self.assertEqual(type(result), _np.matrix)
            self.assertEqual(result.shape, self.data_shape)


            curr = _np.linalg.norm(result)
            """float: Holds the norm of the newly generated random matrix."""

            # Current norm has virtually no chance of being equal to zero.
            self.assertGreater(curr, self.cutoff_zero)

            # The norm of this iteration's result has virutally no chance of
            # being equal to that of the previous one.
            self.assertNotEqual(prev, curr)
            prev = curr

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: Linear algebra utilities testing suite."""
