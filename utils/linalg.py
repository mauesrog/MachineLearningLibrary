"""Linear Algebra Utilities.

Defines common linear algebra routines frequently used by learning models.

Attributes:
    See `config.linalg`.

"""
import numpy as _np

from config import linalg_defaults
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError
from general import compose
from stats import validate_feature_set


DEFAULT_MAX_RANDOM_VALUE = linalg_defaults["max_random_value"]
DEFAULT_MIN_RANDOM_VALUE = linalg_defaults["min_random_value"]


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

def factors(n):
    """Factor Finder.

    Computes all the factors of the given number.

    See:
        https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python

    Args:
        n (int): Number whose factors want to be found.

    Returns:
        list of int: `n`'s factors (if any).

    Raises:
        TypeError: If `n` is not an integer.

    """
    if type(n) != int:
        raise TypeError("Expected 'int', saw '%s' instead." % type(n).__name__)

    candidates = range(1, int(n ** 0.5) + 1)
    """list of int: All integers within the upper bound given by the square root
    of `n`."""
    factors = [[i, n // i] for i in candidates if n % i == 0]
    """list of int: `n`'s factors with possible duplicates for perfect
    squares."""

    try:
        return compose(list, set, reduce)(list.__add__, factors)
    except TypeError:
        raise ValueError("Factors of input '%s' are not computable." % str(n))

def random_matrix(size, min_val=DEFAULT_MIN_RANDOM_VALUE,
                  max_val=DEFAULT_MAX_RANDOM_VALUE):
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

    Appends the given row/column vector(s) at the specified position of a
    matrix.

    Args:
        X (np.matrix): Feature set to be augmented.
        v (np.matrix): Vector(s) (as a matrix).
        position (str): 'bottom', 'left', 'right', or 'top'.

    Returns:
        np.matrix: The augmented feature set.

    Raises:
        IncompatibleDataSetsError: If the feature set's and vector's dimensions
            do not match.
        InvalidFeatureSetError: If the given feature set or vector are invalid.

    """
    map(validate_feature_set, [X, v])

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

    if X.shape[index] != v.shape[index]:
        raise _IncompatibleDataSetsError(X, v, "concatenation")

    return compose(_np.matrix, _np.concatenate)(*args)
