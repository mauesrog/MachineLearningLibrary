"""Loss Functions Testing Module.

"""
import numpy as _np
import unittest as _unittest
from random import uniform as _uniform

from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidObservationSetError as _InvalidObservationSetError
from linalg import random_matrix as _random_matrix
from loss import mse 

class _Test(_ModuleTestCase):
    """Loss Function Unit Tester.

    Runs tests for all loss functions:
        - `mse`

    Attributes:
        cutoff_zero (float): The largest value treated as zero in all equality
            tests.
        label (str): Identifier for super class to generate custome test
            docstrings according to the loss function module.
        max_mean (float): Max MSE to coerce for random tests.
        n (int): Number of data points to use in any test e.g. in random tests.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.

    """
    def setUp(self):
        """Loss Function Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.cutoff_zero = 1e-2
        self.label = '`loss`'
        self.max_mean = 1e6
        self.n = 100
        self.n_tests = 20
        self.name = __name__

    def test_edge_cases_mse(self):
        """`loss.mse`: Edge Case Validator.

        Tests the behavior of `mse` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidObservationSetError):
            # Empty observation matrix.
            mse(_np.matrix([[]]), _random_matrix((self.n, 1)))

        with self.assertRaises(_InvalidObservationSetError):
            # Empty prediction matrix.
            mse(_random_matrix((self.n, 1)), _np.matrix([[]]))

    def test_invalid_args_mse(self):
        """`loss.mse`: Argument Validator.

        Tests the behavior of `mse` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            mse()

        with self.assertRaises(TypeError):
            # Only one arguments.
            mse(123)

        with self.assertRaises(TypeError):
            # More than two arguments.
            mse(123, 123, 123, 1123)

        with self.assertRaises(_InvalidObservationSetError):
            # List of an empty list instead of observation matrix.
            mse([[]], _random_matrix((self.n, 1)))

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of prediction matrix.
            mse(_random_matrix((self.n, 1)), [[]])

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible observation and prediction matrices.
            mse(_random_matrix((self.n, 1)), _random_matrix((self.n + 1, 1)))

    def test_random_mse(self):
        """`loss.mse`: Randomized Validator.

        Tests the behavior of `mse` by feeding it randomly generated arguments.

        Raises:
            AssertionError: If `mse` needs debugging.

        """
        for i in range(0, self.n_tests):
            Y = _random_matrix((self.n, 1), max_val=self.max_mean)
            """float: Random-valued observations."""
            Y_hat = _random_matrix((self.n, 1), max_val=self.max_mean)
            """float: Random-valued predictions."""
            delta_Y = abs(Y - Y_hat)
            """float: Distance between predictions and observations."""
            squared_sum_delta_Y = _np.linalg.norm(delta_Y[1:, 0]) ** 2
            """float: Sum of the squares of all `delta_Y` values."""

            # To ensure that the coercion does not result in the square of a
            # negative number, we can use the mean of the upper-bound
            # `squared_sum_delta_Y` as insurance that the computation will only
            # work with positive numbers.
            err = _uniform((squared_sum_delta_Y + 1.0) / self.n,
                           (squared_sum_delta_Y + self.max_mean) / self.n)
            """float: MSE to coerce."""

            # Coerce MSE by changing the first prediction to a strategic choice
            # and mathematically guaranteeing
            Y_hat[0, 0] = (_np.sqrt(self.n * err - squared_sum_delta_Y) -
                           Y[0, 0]) * -1.0

            result = mse(Y, Y_hat)
            """float: Test input."""

            self.assertLessEqual(abs(result - err), self.cutoff_zero)

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: Loss function testing suite."""
