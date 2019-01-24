"""Augmentors.

Defines functions that augment feature sets.

"""
import numpy as _np
import math as _math
import unittest as _unittest

from linalg import append_left as _append_left, random_matrix as _random_matrix
from test import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError


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
    if type(X) != _np.matrix:
        raise _InvalidFeatureSetError(X, isType=True)

    if X.size == 0:
        reason = "Cannot augment empty matrix!"
        raise _InvalidFeatureSetError(X, reason=reason)

    v = _np.matrix(_np.ones((X.shape[0], 1)))
    """np.matrix: Unit-valued vector to append to feature set."""

    return _append_left(X, v)


class _Test(_ModuleTestCase):
    """Augmentors Unit Tester.

    Runs tests for all general model utilties:
        - `constant_augmentor`

    Attributes:
        cutoff_zero (float): The largest value treated as zero in all equality
            tests.
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the linear model module.
        model (LinearModel): Instance of `LinearModel`.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.
        shapes (tuple of (int, int)): Dimensions of all matrix parameters.

    """
    def setUp(self):
        """Linear Model Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.cutoff_zero = 1e-2
        self.data_shape = 100, 20
        self.label = '`augmentor`'
        self.n_tests = 20
        self.name = __name__

    def test_edge_cases_constant_augmentor(self):
        """`augmentors.constant_augmentor`: Edge Case Validator.

        Tests the behavior of `constant_augmentor` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix `X`.
            constant_augmentor(_np.matrix([[]]))

    def test_invalid_args_constant_augmentor(self):
        """`augmentors.constant_augmentor`: Argument Validator.

        Tests the behavior of `constant_augmentor` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            constant_augmentor()

        with self.assertRaises(TypeError):
            # Too many arguments.
            constant_augmentor(123, 123)

        with self.assertRaises(TypeError):
            # With **kwargs.
            constant_augmentor(_random_matrix((self.data_shape)), key1="key1")

        with self.assertRaises(_InvalidFeatureSetError):
            # String instead of matrix `X`.
            constant_augmentor("string")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            constant_augmentor(None)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of matrix `X`.
            constant_augmentor(_np.zeros(self.data_shape))

    def test_random_model_constant_augmentor(self):
        """`augmentors.constant_augmentor`: Randomized Validator.

        Tests the behavior of `constant_augmentor` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `constant_augmentor` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""
            n = X.shape[0]
            """int: Number of data points."""

            new_X = constant_augmentor(X)
            """np.matrix: Test input."""

            # Augmentation should also be a matrix.
            self.assertEqual(type(X), _np.matrix)

            # Unit-valued vector should have been appended to the left of `X`.
            self.assertEqual(new_X.shape[1], X.shape[1] + 1)

            # Yet the number of rows should remain the same.
            self.assertEqual(new_X.shape[0], X.shape[0])

            # Yet the number of rows should remain the same.
            self.assertEqual(new_X.shape[0], X.shape[0])

            # The norm of the leftmost column vector of `new_X` should be
            # computable accordin to the following formula.
            self.assertLessEqual(abs(_np.linalg.norm(new_X[:, 0]) -
                                                     _math.sqrt(n)),
                                 self.cutoff_zero)

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: General utilities testing suite."""
