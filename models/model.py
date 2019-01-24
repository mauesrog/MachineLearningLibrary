"""Model.

This module describes an abstraction of a learning model for specific
mathematical models to implement.

Attributes:
    eps (float): Step size of all numerical gradients.

"""

from copy import deepcopy as _deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np

from utils import loss as losses
from utils.general import compose
from utils.linalg import diagonal, random_matrix
from test import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompleteModelError as _IncompleteModelError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidModelParametersError as _InvalidModelParametersError, \
                              InvalidObservationSetError as _InvalidObservationSetError

eps = 1e-6
DEFAULT_SHAPE_GRADIENT_CHECKER = (10, 200)


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
        return X

    def evaluate(self, X, Y, loss_fn="mse", regularize=True, **kwargs):
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
            **kwargs: Additional parameters needed by the specific model to
                predict observations based on the given feature set.

        Returns:
            (float, np.matrix): The evaluation error along with the predicted
                observations.

        Raises:
            AttributeError: If `loss_fn` is not recognized.

        """
        if not hasattr(losses, loss_fn):
            raise AttributeError("Unknown loss function '%s'." % loss_fn)

        loss = getattr(losses, loss_fn)
        """callable: Loss function."""

        Y_hat = self.predict(X, **kwargs)
        """np.matrix: Observation predictions."""

        eval = loss(Y, Y_hat)
        """float: Evaluation error."""

        # Only consider regularization if needed e.g. if parameters are being
        # trained.
        if regularize:
            n = X.shape[0]
            """int: Number of data points."""

            # Revert mean computations and add regularization.
            eval *= n
            eval += self.regularization(**kwargs)

        return eval, Y_hat

    def gradient_checker(self, perturbations,
                         shape=DEFAULT_SHAPE_GRADIENT_CHECKER):
        """Model Gradient Validator.

        Checks the accuracy of the analytical computation of all gradients
        needed by the model by juxtaposing the norms of all numerical gradients
        with those of the analytical gradient.

        Args:
            perturbations (int): Number of comparison points to consider in
                norm computation.
            shape ((int, int), optional): Number of data points and number of
                features. Defaults to `DEFAULT_SHAPE_GRADIENT_CHECKER`.

        Returns:
            (list of float, list of float): Norms for all analytical and
                numerical gradients, respectively.

        """
        n, d = shape
        """(int, int): Number of data points and number of features."""
        grad_norms = [0.0 for i in range(0, perturbations)]
        """list of float: Norms of all analytical gradients."""
        ngrad_norms = [0.0 for i in range(0, perturbations)]
        """list of float: Norms of all numerical gradients."""

        # Delete parameters and create random new ones with the specified number
        # of features.
        del self.params
        self.train(np.matrix(np.zeros((1, d))), None, exact=False)

        for i in range(0, perturbations):
            X = random_matrix((n, d))
            """np.matrix: Random-valued feature set."""
            Y = random_matrix((n, 1))
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


    def numerical_gradient(self, X, Y, params=None, **kwargs):
        """Model Numerical Differentiator.

        Given sets of features and observations, computes the numerical gradient
        of the model parameters according to the module attribute `eps`.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            params (tuple of np.matrix, optional): Parameters to use for
                gradient computation. Defaults to `None`, which implies that the
                current trained parameters (i.e. `self.params`) will be used.
            **kwargs: Additional parameters needed by the specific model to
                predict observations based on the given feature set.

        Returns:
            tuple of np.matrix: Parameter numerical gradients.

        """
        if not params:
            params = self.params

        if not params:
            raise _IncompleteModelError("Model parameters not set.")

        grads = map(np.zeros, map(lambda p: p.shape, params))
        """list of (int, int): Matrix dimensions for all parameters."""
        err1 = self.evaluate(X, Y, params=params, **kwargs)[0]
        """float: Model loss at the given parameters."""

        for t in range(0, len(params)):
            param = params[t]
            """np.matrix: Parameter to differentiate in current iteration."""
            grad = grads[t]
            """np.matrix: `param`'s numerical gradient."""
            n, m = grad.shape
            """np.matrix: Current parameter's matrix dimensions."""

            for i in range(0, n):
                for j in range(0, m):
                    param[i, j] += eps  # Increase current partial derivative.

                    err2 = self.evaluate(X, Y, params=params, **kwargs)[0]
                    grad[i, j] = (err2 - err1) / eps  # Numerical gradient.

                    param[i, j] -= eps  # Revert to initial value.

        return tuple(grads)

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
        if not params:
            params = self.params

        if type(params) != tuple:
            raise _InvalidModelParametersError(params)

        for p in params:
            if type(p) != np.matrix or p.size == 0:
                raise _InvalidModelParametersError(params)

        r = self._regularization
        """float: L2 regularization constant."""

        left_multiply = lambda p: compose(p.T.dot, diagonal)(p.shape[0], r)
        """callable: Left multiplies a matrix with the specified diagonal
        matrix."""

        penalizer = lambda p: float(left_multiply(p).dot(p))
        """callable: Given a parameter, computes it's L2 penalty."""

        return compose(sum, map)(penalizer, params)

    @abstractmethod
    def gradient(self, X, Y, **kwargs):
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
    def predict(self, X, **kwargs):
        """Model Predictor.

        Given a set of features, predicts observations based on the current
        trained parameters or on the given arbitrary ones.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            **kwargs: Additional parameters needed by the specific model to
                predict observations based on the given feature set. Parameters
                specified with the key 'params' will replace the default trained
                parameters saved by the model.

        Returns:
            np.matrix: Predicted observations derived from features.

        Raises:
            ValueError: If no valid model parameters are available.

        """
        pass

    @abstractmethod
    def train(self, X, Y, **kwargs):
        """Model Trainer.

        Uses the given sets of features and observations to compute the global
        minimum of the loss function, thus computing optimally trained
        parameters.

        Note:
            This method is unfeasible with large datasets, it is virtually for
            testing purposes only.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            **kwargs: Additional parameters needed to train the specific model.

        Returns:
            float: Training error.

        """
        pass


class _Test(_ModuleTestCase):
    """Model Unit Tester.

    Runs tests for all common properties and methods of model implementations:
        - `params`

    """
    def tearDown(self):
        """Model Testing Destructor.

        Cleans up of after running each specific model test.

        """
        del self.model.params

        # Model parameters should be uninitialized after deletion.
        self.assertIsNone(self.model.params)

    def test_edge_cases_model_params(self):
        """`Model.params`: Edge Case Validator.

        Tests the behavior of `params` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidModelParametersError):
            # Empty matrix instead of list of matrices `new_params`.
            self.model.params = [np.matrix([[]])]

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

    def test_edge_cases_model_augment(self):
        """`Model.augment`: Edge Case Validator.

        Tests the behavior of `augment` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix instead of matrix `X`.
            self.model.augment(np.matrix([[]]))

    def test_edge_cases_model_regularization(self):
        """`Model.regularization`: Edge Case Validator.

        Tests the behavior of `regularization` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidModelParametersError):
            # Tuple of empty matrices.
            self.model.regularization((np.matrix([[]]), np.matrix([[]])))

    def test_invalid_args_model_params(self):
        """`Model.params`: Argument Validator.

        Tests the behavior of `params` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(_InvalidModelParametersError):
            params = map(random_matrix, self.shapes)
            """list of np.matrix: Zero-valued parameters."""

            # Insert extra parameter.
            params.append(params[0])

            # More parameters than expected.
            self.model.params = params

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

        with self.assertRaises(_InvalidModelParametersError):
            # `None` instead of list of matrices `new_params`.
            self.model.params = None

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

        with self.assertRaises(_InvalidModelParametersError):
            # Matrix instead of list of matrices `new_params`.
            self.model.params = np.matrix((n, d))

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

        with self.assertRaises(_InvalidModelParametersError):
            # Tuple instead of list of matrices `new_params`.
            self.model.params = np.matrix((n, d)), np.matrix((n, d))

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

        with self.assertRaises(_InvalidModelParametersError):
            # List of empty lists instead of list of matrices `new_params`.
            self.model.params = [[], [], []]

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

        with self.assertRaises(_InvalidModelParametersError):
            # List of empty list instead of list of matrices `new_params`.
            self.model.params = [[]]

        # Model parameters should still be uninitialized.
        self.assertIsNone(self.model.params)

    def test_invalid_args_model_augment(self):
        """`Model.augment`: Argument Validator.

        Tests the behavior of `augment` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.augment()

        with self.assertRaises(TypeError):
            # More parameters than expected.
            self.model.augment(12, 12)

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of matrix `X`.
            self.model.augment(None)

        with self.assertRaises(_InvalidFeatureSetError):
            # Integer instead of matrix `X`.
            self.model.augment(123)

        with self.assertRaises(_InvalidFeatureSetError):
            # 1-Tuple of a matrix instead of matrix `X`.
            self.model.augment((random_matrix((n, d)),))

        with self.assertRaises(_InvalidFeatureSetError):
            # List of an empty list instead of matrix `X`.
            self.model.augment([[]])

        with self.assertRaises(_InvalidFeatureSetError):
            # Array instead of matrix `X`.
            self.model.augment(np.ndarray(n))

    def test_invalid_args_model_regularization(self):
        """`Model.regularization`: Argument Validator.

        Tests the behavior of `regularization` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidModelParametersError):
            # No arguments with no parameters set.
            self.model.regularization()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.regularization(params, params)

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.regularization(params, params=params)

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            self.model.regularization(params=params, key="value")

        with self.assertRaises(_InvalidModelParametersError):
            # List instead of parameter tuple `params`.
            self.model.regularization(list(params))

        with self.assertRaises(_InvalidModelParametersError):
            # Empty list tuple instead of parameter tuple `params`.
            self.model.regularization(params=([], []))

    def test_random_model_params(self):
        """`Model.params`: Randomized Validator.

        Tests the behavior of `params` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `params` needs debugging.

        """
        for i in range(0, self.n_tests):
            random_params = compose(tuple, map)(random_matrix, self.shapes)
            """list of np.matrix: Randomized set of parameters."""

            self.model.params = random_params

            params = self.model.params
            """list of np.matrix: Deep copy of newly set parameters."""

            # Parameters should be a tuple.
            self.assertEqual(type(params), tuple)

            # Number of parameters should match number of parameter dimensions.
            self.assertEqual(len(params), len(self.shapes))

            for j in range(len(params)):
                # Each parameter should be a matrix.
                self.assertEqual(type(params[j]), np.matrix)

                # Each parameter from input should match the correspoding
                # parameter copied with the getter method.
                self.assertEqual(np.linalg.norm(params[j]),
                                 np.linalg.norm(random_params[j]))

            # Model parameters should be initialized at this point.
            self.assertIsNotNone(self.model.params)

            del self.model.params

            # Model parameters should be uninitialized after deletion.
            self.assertIsNone(self.model.params)

    def test_random_model_augment(self):
        """`Model.augment`: Randomized Validator.

        Tests the behavior of `augment` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `augment` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""

            new_X = self.model.augment(X)
            """np.matrix: Test input."""

            # Augmentation should also be a matrix.
            self.assertEqual(type(X), np.matrix)

            # Total number of values in augmented matrix should be greter than
            # or equal to the number of values in the original matrix.
            if new_X.shape[0] != X.shape[0] or new_X.shape[1] != X.shape[1]:
                self.assertGreaterEqual(new_X.shape[0], X.shape[0])
                self.assertGreaterEqual(new_X.shape[1], X.shape[1])

    def test_random_model_regularization(self):
        """`Model.regularization`: Randomized Validator.

        Tests the behavior of `regularization` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `regularization` needs debugging.

        """
        for i in range(0, self.n_tests):
            random_params = compose(tuple, map)(random_matrix, self.shapes)
            """tuple of np.matrix: Random-valued parameters."""

            # First, test `params` as a method argument.
            result = self.model.regularization(random_params)
            """float: Test input."""

            self.assertEqual(type(result), float)

            # Finally, test `params` as attribute.
            self.model.params = random_params

            result = self.model.regularization()

            self.assertEqual(type(result), float)
