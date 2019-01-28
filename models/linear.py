"""Linear Model.

Implements an L2-regularized learning model exclusively capable of finding
linear solutions.

"""
import numpy as _np

from model import Model as _Model
from utils.augmentors import constant_augmentor as  _constant_augmentor


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
        new_X = _constant_augmentor(X)
        """np.matrix: Augmented feature set."""

        return super(LinearModel, self).augment(new_X)

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
