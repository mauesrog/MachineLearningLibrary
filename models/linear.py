"""Linear Model.

Implements an L2-regularized learning model exclusively capable of finding
linear solutions.

"""
import numpy as np

from config import model_defaults
from model import Model
from utils.augmentors import constant_augmentor
from utils.linalg import diagonal, random_matrix
from common.exceptions import InvalidModelParametersError

DEFAULT_REGULARIZATION = model_defaults["regularization"]


class LinearModel(Model):
    """Linear Model.

    Implements abstract class `Model` to perform learning operations according
    to the standard linear regression model.

    Note:
        See base class for attribute inheritance details. Only class-specific
        attributes are listed here.

    Attributes:
        _a (np.matrix): Linear weights.

    """
    def __init__(self, regularization=DEFAULT_REGULARIZATION):
        """Linear Model Constructor.

        Args:
            regularization (float): See `self._regularization` in base class
                attributes.

        """
        super(LinearModel, self).__init__(regularization, '_a')

    def augment(self, X):
        """Enables linear combinations with constants by adding a column of
        unit-valued features to the given feature set."""
        new_X = constant_augmentor(X)
        """np.matrix: Augmented feature set."""

        return super(LinearModel, self).augment(new_X)

    def gradient(self, X, Y, params=None):
        """See base class."""
        def action():
            """Gradient Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                tuple of np.matrix: Parameter gradients.

            """
            n, d = X.shape
            """(int, int): Number of data points and number of features."""

            Y_hat = self.predict(X)
            """np.matrix: Observation predictions."""

            delta_Y = Y - Y_hat
            """np.matrix: Difference between observations and predictions."""

            return (-2.0 * (X.T.dot(delta_Y) - self._regularization * self._a),)

        return super(LinearModel,
                     self)._update_model(action, X=X, Y=Y, params=params)

    def init_params(self, X):
        """See base class.

        Args:
            X (np.matrix): Feature set. Shape: n x d.

        """
        shape_fn = lambda X: ((X.shape[1], 1),)
        """callable: Returns matrix dimensions from feature sets."""

        super(LinearModel, self).init_params(X, shape_fn)

    def predict(self, X, params=None):
        """See base class."""
        def action():
            """Predict Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                np.matrix: Predicted observations.

            Raises:
                InvalidModelParametersError: If the parameters provided are not
                    compatible with the given feature set.

            """
            try:
                return X.dot(self._a)
            except ValueError:
                reason = "Linear weights' size %s does not match feature set " \
                         "size %s." % (self._a.shape, X.shape)
                raise InvalidModelParametersError(self.params, reason=reason)

        return super(LinearModel,
                     self)._update_model(action, X=X, params=params)

    def train(self, X, Y):
        """See base class."""
        def action():
            """Train Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                float: The evaluation error along with the predicted
                    observations.

            """
            reg = diagonal(X.shape[1], self._regularization)
            """np.matrix: Diagonal L2 regularization matrix"""

            self.params = (np.matrix(X.T.dot(X) + reg).I.dot(X.T).dot(Y),)

            return self.evaluate(X, Y)[0]

        return super(LinearModel, self)._update_model(action, X=X, Y=Y,
                                                      no_params=True)
