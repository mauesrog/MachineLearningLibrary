"""Augmentors Testing Module.

Attributes:
    Test (TestSuite): Augmentators testing suite.

"""
import numpy as _np
import math as _math
import unittest as _unittest

from linalg import random_matrix as _random_matrix
from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError
from augmentors import constant_augmentor


class _Test(_ModuleTestCase):
    """Augmentors Unit Tester.

    Runs tests for all general model utilties:
        - `constant_augmentor`

    Attributes:
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
        for i in range(self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""
            n = X.shape[0]
            """int: Number of data points."""

            new_X = constant_augmentor(X)
            """np.matrix: Test input."""

            # Augmentation should also be a matrix.
            self.assertIsInstance(X, _np.matrix)

            # Unit-valued vector should have been appended to the left of `X`.
            self.assertEqual(new_X.shape[1], X.shape[1] + 1)

            # Yet the number of rows should remain the same.
            self.assertEqual(new_X.shape[0], X.shape[0])

            # Yet the number of rows should remain the same.
            self.assertEqual(new_X.shape[0], X.shape[0])

            # The norm of the leftmost column vector of `new_X` should be
            # computable accordin to the following formula.
            self.assertAlmostEqual(_np.linalg.norm(new_X[:, 0]), _math.sqrt(n))

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
