"""Linear Model Testing Module.

"""
import numpy as _np
import math as _math
import unittest as _unittest

from common.test_cases.model_test_case import ModelTestCase as _ModelTest
from linear import LinearModel as _LinearModel
from utils.linalg import random_matrix as  _random_matrix


class _Test(_ModelTest):
    """Linear Model Unit Tester.

    Runs tests for all properties and methods of `Model`, plus those particular
    to `LinearModel`:
        - `augment`
        - `gradient`
        - `predict`
        - `train`

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
        self.label = '`linear.LinearModel`'
        self.model = _LinearModel(0.5)
        self.n_tests = 20
        self.name = __name__
        self.shapes = tuple([(20, 1)])

        # Model parameters should be uninitialized at this point.
        self.assertIsNone(self.model.params)

    def test_random_model_augment(self):
        """`LinearModel.augment`: Randomized Validator.

        Tests the behavior of `augment` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `augment` needs debugging.

        """
        # Run `Model`-wide `augment` unit tests.
        super(_Test, self).test_random_model_augment()

        for i in range(0, self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""
            n = X.shape[0]
            """int: Number of data points."""

            new_X = self.model.augment(X)
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
"""TestSuite: Linear model testing suite."""
