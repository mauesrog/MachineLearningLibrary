"""Linear Model.

Implements an L2-regularized learning model exclusively capable of finding
linear solutions.

"""
import numpy as _np
import math as _math
import unittest as _unittest

from model import Model as _Model, _Test as _ModelTest
from utils.augmentors import constant_augmentor as  _constant_augmentor
from utils.linalg import random_matrix as  _random_matrix


class LinearModel(_Model):
    """Linear Model.

    Implements abstract class `Model` to perform learning operations according
    to the standard linear regression model.

    Note:
        See base class for attribute inheritance details. Only class-specific
        attributes are listed here.

    Attributes:
        _a (np.matrix): Linear weigths.

    """
    def __init__(self, regularization):
        """Linear Model Constructor.

        Args:
            regularization (float): See `self._regularization` in base class
                attributes.

        """
        super(LinearModel, self).__init__(regularization, '_a')

    def augment(self, X):
        """Enables linear combinations with constants by adding a column of
        unit-valued features to the given feature set."""
        return _constant_augmentor(X)

    def gradient(self, X, Y, params=None):
        """See base class."""
        a = self._a if params is None else params[0]
        """np.matrix: Linear weights."""

        if a is None:
            raise ValueError('Cannot compute gradient with no linear weights set!')

        n, d = X.shape
        Y_hat = self.predict(X, params=[a])

        grad = -2.0 * (X.T.dot(Y - Y_hat) - self._regularization * a)

        return [grad]

    def predict(self, X, params=None):
        """See base class."""
        a = self._a if params is None else params[0]
        """np.matrix: Linear weights."""

        if a is None:
            raise ValueError('Cannot predict with no linear weights set!')

        return _np.matrix(X.dot(a))

    def train(self, X, Y, exact=False, params=None):
        """See base class."""
        n, d = X.shape

        if not exact:
            self._a = _np.random.rand(d, 1)
            return None

        self._a = _np.matrix(X.T.dot(X) + self._regularization * _np.identity(d)).I.dot(X.T).dot(Y)

        return self.evaluate(X, Y)[0]


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
        self.model = LinearModel(0.5)
        self.n_tests = 20
        self.name = __name__
        self.shapes = tuple([(100, 1)])

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
