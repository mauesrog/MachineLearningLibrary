"""Linear Algebra Utilities.

Defines common linear algebra routines frequently used by learning models.

"""
import math as _math
import numpy as _np
import unittest as _unittest

from test import ModuleTestCase as _ModuleTestCase

from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError

from general import compose

def append_bottom(X, v):
    """Matrix Bottom Concatenator.

    Appends the given column vector to the bottom of a matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Column vector (as a matrix).

    Returns:
        np.matrix: The augmented feature set.

    """
    return _append_helper(X, v, "bottom")

def append_left(X, v):
    """Matrix Left Concatenator.

    Appends the given column vector to the left of a matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Column vector (as a matrix).

    Returns:
        np.matrix: The augmented feature set.

    """
    return _append_helper(X, v, "left")

def append_right(X, v):
    """Matrix Right Concatenator.

    Appends the given column vector to the right of a matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Column vector (as a matrix).

    Returns:
        np.matrix: The augmented feature set.

    """
    return _append_helper(X, v, "right")

def append_top(X, v):
    """Matrix Top Concatenator.

    Appends the given column vector to the top of a matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Column vector (as a matrix).

    Returns:
        np.matrix: The augmented feature set.

    """
    return _append_helper(X, v, "top")

def diagonal(n, val=1.0):
    """Diagonal Matrix Generator.

    Creates a matrix having non-zero elements only in the diagonal running from
    the upper left to the lower right corner.

    Args:
        n (int): Diagonal length.
        val (float, optional): Value to use for all non-zero elements. Defaults
            to the identity matrix.

    Returns:
        np.matrix: Diagonal matrix.

    Raises:
        TypeError: If `val` is neither a float nor an integer.
        ValueError: If `n` is less than or equal to zero.

    """
    if n <= 0:
        raise ValueError("Expected natural number, saw '%d' instead." % n)

    if type(val) != int and type(val) != float:
        raise TypeError("Expected 'int' or 'float', saw '%s' instead." %
                        type(val).__name__)

    # Multiplying the identity element of the given matrix space by the
    # specified constant does the trick.
    return _np.matrix(val * _np.identity(n))

def random_matrix(size, min_val=0.0, max_val=1.0):
    """Random Matrix Generator.

    Creates a matrix of the given size with values determined by the specified
    bounds.

    Args:
        size ((int, int)): Matrix dimensions.
        min_val (float, optional): Smallest random float allowed. Defaults to
            0.0.
        max_val (float, optional): Largest random float allowed. Defaults to
            1.0.

    Returns:
        np.matrix: Random matrix.

    Raises:
        TypeError: If the given size is not a tuple, or if the given bounds are
            not numbers.
        ValueError: If the given size is not a 2-tuple of natural numbers, or
            if the given bounds are incompatible.

    """
    if type(size) != tuple:
        raise TypeError("Expected 'tuple', saw '%s' instead." %
                        type(size).__name__)

    if len(size) != 2:
        raise ValueError("Only two-dimensional matrices supported.")

    if (type(size[0]) != int or type(size[0]) != int) or \
       (size[0] <= 0 or size[1] <= 0):
        raise ValueError("Dimensions need to be natural numbers.")

    if type(min_val) != int and type(min_val) != float:
        raise TypeError("Expected 'int' or 'float', saw '%s' instead." %
                        type(min_val).__name__)

    if type(max_val) != int and type(max_val) != float:
        raise TypeError("Expected 'int' or 'float', saw '%s' instead." %
                        type(min_val).__name__)

    if not min_val < max_val:
        raise ValueError("Incompatible bounds: [%.2f, %.2f)." % (min_val,
                                                                 max_val))


    return _np.matrix(_np.random.uniform(min_val, max_val, size=size))

def _append_helper(X, v, position):
    """Matrix Concatenator.

    Appends the given row/column vector at the specified position of a matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Vector (as a matrix).
        position (str): 'bottom', 'left', 'right', or 'top'.

    Returns:
        np.matrix: The augmented feature set.

    Raises:
        IncompatibleDataSetsError: If the feature set's and vector's dimensions
            do not match or if `v` is not unit-dimensioned.
        InvalidFeatureSetError: If the given feature set or vector are invalid.

    """
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if X.size == 0:
        raise _InvalidFeatureSetError(X, reason="Cannot augment empty matrix!")

    if type(v) != _np.matrix:
        raise _InvalidFeatureSetError(v, isType=True)

    if v.size == 0:
        raise _InvalidFeatureSetError(v, reason="Received empty column vector.")

    switcher=dict(bottom=((X, v), 0), left=((v, X), 1), right=((X, v), 1),
                  top=((v, X), 0))
    """:obj:`((np.matrix, np.matrix), int)`: Helper to determine how to
    concatenate the vector and the matrix."""

    args = switcher.get(position)
    """((np.matrix, np.matrix), int): The explicit matrix pair to determine the
    order of concatenation and values 0 or 1 to determine its direction. See
    `np.concatenate`."""

    if not args:
        raise ValueError("Invalid append position.")

    index = int(position == 'bottom' or position == 'top')
    """int: Index into both the matrix's and the vector's `shape` attributes.
    Determines which dimensions should be aligned."""

    if X.shape[index] != v.shape[index] or v.shape[(index + 1) % 2] != 1:
        raise _IncompatibleDataSetsError(X, v, "concatenation")

    return compose(_np.matrix, _np.concatenate)(*args)


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
