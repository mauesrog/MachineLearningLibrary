"""Model.

This module describes an abstraction of a learning model for specific
mathematical models to implement.

Attributes:
    See `config.models`.

"""

from copy import deepcopy as _deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

from config import model_defaults
from utils import loss as losses
from utils.stats import validate_datasets, validate_feature_set, \
                        validate_observation_set
from utils.general import compose
from utils.linalg import diagonal, random_matrix
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              IncompleteModelError as _IncompleteModelError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidModelParametersError as _InvalidModelParametersError, \
                              InvalidObservationSetError as _InvalidObservationSetError

NUM_GRAD_EPS = model_defaults["num_grad_eps"]
DEFAULT_SHAPE_CHECKER = model_defaults["gradient_checker_shape"]


class Model(object):
    """Model Abstract Class.

    Other than abstracting certain logic from specific models, the module also
    defines an interface that all model compatible with learning should
    implement.

    Attributes:
        _param_names (list of str): Names of all model parameters.
        _regularization (float): L2-regularization constant.

    """
    __metaclass__ = ABCMeta

    def __init__(self, regularization, *param_names):
        """Model Constructor.

        Args:
            See class attributes.

        Raises:
            AttributeError: If no parameter names are provided.

        """
        if not len(param_names):
            raise AttributeError('All models need parameters.')

        self._param_names = param_names
        self._regularization = regularization

        # Initialize all parameters to `None`.
        del self.params

    def __str__(self):
        param_to_str = (lambda name:
            "`%s`: %s" % (name[1:], "not set"
                                    if not hasattr(self, name) or getattr(self, name) is None
                                    else getattr(self, name).shape)
        )
        """callable: Stringifies given parameter."""

        param_str = compose(", ".join, map)(param_to_str, self._param_names)
        """str: Stringified parameters."""

        return "%s: %s" % (self.__class__.__name__, param_str)

    @property
    def params(self):
        """tuple of np.matrix: Deep copy of current trained parameters."""
        params = map(lambda name: _deepcopy(getattr(self, name)),
                     self._param_names)
        """list of np.matrix: List of all found parameters."""

        # Return `None` if any parameter is `None`.
        for p in params:
            if p is None:
                return None

        return tuple(params)

    @params.setter
    def params(self, new_params):
        """The new parameters provided need to match the conditions set by
        `self._param_names`.

        Raises:
            InvalidModelParametersError: Any of these reasons:
                - `new_params` is not a tuple.
                - Provided less or more than the number of parameters
                  expected.
                - `None`-type parameter encountered.
                - Empty parameter encountered.
        """
        # Validate `new_params` type and length.
        if type(new_params) != tuple:
            raise _InvalidModelParametersError(new_params)

        if len(new_params) != len(self._param_names):
            reason = "Expected exactly %d parameters, saw %d instead." % \
                     (len(self._param_names), len(new_params))
            raise _InvalidModelParametersError(new_params, reason=reason)

        # Ensure that all parameters are non-empty, real-valued matrices.
        for p in new_params:
            if type(p) != np.matrix:
                raise _InvalidModelParametersError(new_params)

            if p.size == 0:
                reason = "Empty parameters are not allowed."
                raise _InvalidModelParametersError(new_params, reason=reason)

        # Update each parameter into its corresponding attribute.
        for i in range(0, len(new_params)):
            p = new_params[i]
            """np.matrix: Current model parameter upadate."""
            name = self._param_names[i]
            """str: Key of attribute that should hold current parameter."""

            setattr(self, name, p)

    @params.deleter
    def params(self):
        """Note: Only parameters registered in `self._param_names` will get
        updated."""
        for name in self._param_names:
            setattr(self, name, None)

    def augment(self, X):
        """Feature Set Augmentor.

        Optionally augments the given feature set to suit specific model
        implementations.

        Args:
            X (np.matrix): Feature set. Shape: n x d.

        Returns:
            np.matrix: Augmented feature set.

        """
        action = lambda: X
        """callable: Augment update action."""

        return self._update_model(action, X=X, no_params=True)

    def evaluate(self, X, Y, loss_fn="mse", regularize=True, params=None):
        """Model Evaluator.

        Given a set of features, predicts observations based on the current
        trained parameters and computes their evaluation against the real
        observations.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            loss_fn (str, optional): Loss functions to use. Defaults to 'mse'
                i.e. mean square error.
            regularize (bool, optional): Whether to include L2 regularization
                in evaluation computations. Defaults to `True`.
            params (tuple of np.matrix, optional): Parameters to use for
                evaluation. Defaults to `None`, which implies that the current
                trained parameters (i.e. `self.params`) will be used.

        Returns:
            (float, np.matrix): The evaluation error along with the predicted
                observations.

        Raises:
            AttributeError: If `loss_fn` is not recognized.

        """
        try:
            loss = getattr(losses, loss_fn)
            """callable: Loss function."""
        except AttributeError:
            raise AttributeError("Unknown loss function '%s'." % loss_fn)

        def action():
            """Evaluate Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                (float, np.matrix): The evaluation error along with the
                    predicted observations.

            """
            Y_hat = self.predict(X)
            """np.matrix: Observation predictions."""

            eval = loss(Y, Y_hat)
            """float: Evaluation error."""

            # Only consider regularization if needed e.g. if parameters are
            # being trained.
            if regularize:
                n = X.shape[0]
                """int: Number of data points."""

                # Revert mean computations and add regularization.
                eval *= n
                eval += self.regularization()

            return eval, Y_hat

        return self._update_model(action, X=X, Y=Y, params=params)

    def gradient_checker(self, perturbations, shape=DEFAULT_SHAPE_CHECKER):
        """Model Gradient Validator.

        Checks the accuracy of the analytical computation of all gradients
        needed by the model by juxtaposing the norms of all numerical gradients
        with those of the analytical gradient.

        Args:
            perturbations (int): Number of comparison points to consider in
                norm computation.
            shape ((int, int), optional): Number of data points and number of
                features. Defaults to `DEFAULT_SHAPE_CHECKER`.

        Returns:
            (list of float, list of float): Norms for all analytical and
                numerical gradients, respectively.

        """
        n, d = shape
        """(int, int): Number of data points and number of features."""

        def action():
            """Gradient Checker Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                (list of float, list of float): The evaluation error along with
                    the predicted observations.

            """
            grad_norms = [0.0 for i in range(0, perturbations)]
            """list of float: Norms of all analytical gradients."""
            ngrad_norms = [0.0 for i in range(0, perturbations)]
            """list of float: Norms of all numerical gradients."""

            for i in range(0, perturbations):
                X = random_matrix((n, d), min_val=0.0, max_val=1.0)
                """np.matrix: Random-valued feature set."""
                Y = random_matrix((n, 1), min_val=0.0, max_val=1.0)
                """np.matrix: Random-valued observation set."""

                grads = self.gradient(X, Y)
                """tuple of np.matrix: Analytical gradients."""
                ngrads = self.numerical_gradient(X, Y)
                """tuple of np.matrix: Numberical gradients."""

                # Add gradient norms to norm totals.
                for j in range(0, len(grads)):
                    grad_norms[i] += np.linalg.norm(grads[j]) ** 2
                    ngrad_norms[i] += np.linalg.norm(ngrads[j]) ** 2

            return grad_norms, ngrad_norms

        params = (random_matrix((d, 1), min_val=0.0, max_val=1.0),)
        """tuple of np.matrix: Random-valued parameters."""

        return self._update_model(action, params=params)

    def init_params(self, X, shape_fn):
        """Model Parameter Intializer.

        Given a feature set, intializes all parameters to random values.

        Args:
            shape_fn (callable): Returns matrix dimensions from feature sets.

        """
        def action():
            """Initialize Parameters Update Action.

            Defines the routine to run after the feature sets have been
            validated.

            """
            self.params = compose(tuple, map)(random_matrix, shape_fn(X))

        self._update_model(action, X=X, no_params=True)

    def numerical_gradient(self, X, Y, params=None):
        """Model Numerical Differentiator.

        Given sets of features and observations, computes the numerical gradient
        of the model parameters according to the module attribute
        `NUM_GRAD_EPS`.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            params (tuple of np.matrix, optional): Parameters to use for
                gradient computation. Defaults to `None`, which implies that the
                current trained parameters (i.e. `self.params`) will be used.

        Returns:
            tuple of np.matrix: Parameter numerical gradients.

        """
        def action():
            """Numerical Gradient Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                (float, np.matrix): The evaluation error along with the
                    predicted observations.

            """
            params = self.params
            """tuple of np.matrix: Parameters to use for gradient
            computation."""

            grads = map(random_matrix, map(lambda p: p.shape, params))
            """list of (int, int): Matrix dimensions for all parameters."""
            err1 = self.evaluate(X, Y, params=params)[0]
            """float: Model loss at the given parameters."""

            for t in range(0, len(params)):
                param = params[t]
                """np.matrix: Parameter to differentiate in current
                iteration."""
                grad = grads[t]
                """np.matrix: `param`'s numerical gradient."""
                n, m = grad.shape
                """np.matrix: Current parameter's matrix dimensions."""

                for i in range(0, n):
                    for j in range(0, m):
                        param[i, j] += NUM_GRAD_EPS  # Increase current partial
                                                     # derivative.

                        err2 = self.evaluate(X, Y, params=params)[0]

                        # Numerical gradient.
                        grad[i, j] = (err2 - err1) / NUM_GRAD_EPS

                        param[i, j] -= NUM_GRAD_EPS  # Revert to initial value.

            return tuple(grads)

        return self._update_model(action, X=X, Y=Y, params=params)

    def regularization(self, params=None):
        """Regularization Calculator.

        Computes the L2 penalty for all parameters accroding to the model's
        regularization constant.

        Args:
            params (tuple of np.matrix, optional): Parameters to penalize.
                Defaults to `None`, which implies that the current trained
                parameters (i.e. `self.params`) will be used instead.

        Returns:
            float: Sum of the L2 penalties for all parameters.

        Todo:
            Implement other kinds of regularization e.g. lasso.

        """
        def action():
            """Regularization Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                (float, np.matrix): The evaluation error along with the
                    predicted observations.

            """
            r = self._regularization
            """float: L2 regularization constant."""

            left_multiply = lambda p: compose(p.T.dot, diagonal)(p.shape[0], r)
            """callable: Left multiplies a matrix with the specified diagonal
            matrix."""

            penalizer = lambda p: float(left_multiply(p).dot(p))
            """callable: Given a parameter, computes it's L2 penalty."""

            return compose(sum, map)(penalizer, self.params)

        return self._update_model(action, params=params)

    def _update_model(self, action, **kwargs):
        """Model Update Helper.

        Runs the specified action only after validating the given datasets
        and/or model parameters. Unless the `no_params` flag is set, the model
        parameters will be changed to the given `params` but will get reverted
        as soon as `action` returns.

        Args:
            action (callable): Consequence of successful argument validation.
            **kwargs: Feature set `X`, observation set `Y`, model parameters
                `params`, and/or the flag `no_params`.

        Returns:
            Whatever gets returned by `action`.

        Raises:
            AttributeError: If neither datasets nor parameters were provided.
            IncompleteModelError: If flag `no_params` is inactive and no
                valid parameters could be inferred.

        """
        has_params = "no_params" not in kwargs
        """bool: Whether to look for parameters."""

        if len(kwargs) == 0 or (len(kwargs) == 1 and not has_params):
            raise AttributeError("Update requires at least a feature set and "
                                 "an observation or a parameter tuple.")

        # Validate datasets (if needed)
        if "X" in kwargs and "Y" in kwargs:
            validate_datasets(kwargs["X"], kwargs["Y"])
        else:
            if "X" in kwargs:
                validate_feature_set(kwargs["X"])

            if "Y" in kwargs:
                validate_observation_set(kwargs["Y"])

        # If parameters should be found, infer them from the arguments and
        # current value for `self.params`.
        if has_params:
            old_params = self.params
            """tuple of np.matrix: Reference to current trained parameters."""

            new_params = kwargs["params"] if \
                         "params" in kwargs and kwargs["params"] else \
                         old_params
            """tuple of np.matrix: Parameters to use within `action`."""

            if not new_params:
                raise _IncompleteModelError("Model parameters not set.")

            self.params = new_params

        # Validation was successful, so run given action and track whatever gets
        # returned.
        try:
            result = action()
        except Exception as e:
            if has_params:
                if old_params:
                    self.params = old_params
                else:
                    del self.params

            raise e

        # Revert back to original model parameters.
        if has_params:
            if old_params:
                self.params = old_params
            else:
                del self.params

        return result

    @abstractmethod
    def gradient(self, X, Y, params=None):
        """Model Analytical Differentiator.

        Given sets of features and observations, computes the exact gradient of
        the model parameters.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            **kwargs: Additional parameters needed by the specific model to
                compute the partial derivatives of all parameters. Parameters
                specified with the key 'params' will replace the default trained
                parameters saved by the model.

        Returns:
            tuple of np.matrix: Parameter gradients.

        Raises:
            ValueError: If no valid model parameters are available.

        """
        pass

    @abstractmethod
    def predict(self, X, params=None):
        """Model Predictor.

        Given a set of features, predicts observations based on the current
        trained parameters or on the given arbitrary ones.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            params (tuple of np.matrix, optional): Parameters to use for
                evaluation. Defaults to `None`, which implies that the current
                trained parameters (i.e. `self.params`) will be used.

        Returns:
            np.matrix: Predicted observations derived from features.

        Raises:
            ValueError: If no valid model parameters are available.

        """
        pass

    @abstractmethod
    def train(self, X, Y):
        """Model Trainer.

        Uses the given sets of features and observations to compute the global
        minimum of the loss function, thus computing optimally trained
        parameters.

        Note:
            This method is unfeasible with large datasets, it is basically for
            testing purposes only.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.

        Returns:
            float: Training error.

        """
        pass
