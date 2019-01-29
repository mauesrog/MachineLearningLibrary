"""Model Testing Module.

"""
import numpy as np

from utils.general import compose
from utils.linalg import random_matrix
from module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              IncompleteModelError as _IncompleteModelError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidModelParametersError as _InvalidModelParametersError, \
                              InvalidObservationSetError as _InvalidObservationSetError


class ModelTestCase(_ModuleTestCase):
    """Model Unit Tester.

    Runs tests for all common properties and methods of model implementations:
        - `params`: getter, setter, and deleter
        - `augment`
        - `evaluate`
        - `gradient`
        - `numerical_gradient`
        - `predict`
        - `regularization`
        - `train`

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

    def test_edge_cases_model_evaluate(self):
        """`Model.evaluate`: Edge Case Validator.

        Tests the behavior of `evaluate` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix instead of matrix `X`.
            self.model.evaluate(np.matrix([[]]), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix instead of matrix `Y`.
            self.model.evaluate(X, np.matrix([[]]), params=params)

        with self.assertRaises(_InvalidModelParametersError):
            # Empty matrix instead of matrix `Y`.
            self.model.evaluate(X, Y, params=(np.matrix([[]]),))

    def test_edge_cases_model_gradient(self):
        """`Model.gradient`: Edge Case Validator.

        Tests the behavior of `gradient` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty matrix instead of matrix `X`.
            self.model.gradient(np.matrix([[]]), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty matrix instead of matrix `Y`.
            self.model.gradient(X, np.matrix([[]]), params=params)

        with self.assertRaises(_InvalidModelParametersError):
            # Empty matrix instead of matrix `Y`.
            self.model.gradient(X, Y, params=(np.matrix([[]]),))

    def test_edge_cases_model_numerical_gradient(self):
        """`Model.numerical_gradient`: Edge Case Validator.

        Tests the behavior of `numerical_gradient` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        Y = random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty feature set.
            self.model.numerical_gradient(np.matrix([[]]), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty observation set.
            self.model.numerical_gradient(X, np.matrix([[]]), params=params)

        with self.assertRaises(_InvalidModelParametersError):
            # Empty parameters.
            self.model.numerical_gradient(X, Y, params=(np.matrix([[]]),))

    def test_edge_cases_model_predict(self):
        """`Model.predict`: Edge Case Validator.

        Tests the behavior of `predict` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty feature set.
            self.model.predict(np.matrix([[]]), params=params)

        with self.assertRaises(_InvalidModelParametersError):
            # Empty parameters.
            self.model.predict(X, params=(np.matrix([[]]),))

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

    def test_edge_cases_model_train(self):
        """`Model.train`: Edge Case Validator.

        Tests the behavior of `train` with edge cases.

        Raises:
        Exception: If at least one `Exception` raised is not of the expected
        kind.

        """
        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        Y = random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty feature set.
            self.model.train(np.matrix([[]]), Y)

        with self.assertRaises(_InvalidObservationSetError):
            # Empty observation set.
            self.model.train(X, np.matrix([[]]))

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

    def test_invalid_args_model_evaluate(self):
        """`Model.evaluate`: Argument Validator.

        Tests the behavior of `evaluate` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = random_matrix((n, 1))
        """np.matrix: Random-valued observation set."""
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.evaluate()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.evaluate(X, Y, Y, params=params)

        with self.assertRaises(_IncompleteModelError):
            # Params not set.
            self.model.evaluate(X, Y)

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            self.model.evaluate(X, Y, params=params, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of feature set `X`.
            self.model.evaluate(None, Y, params=params)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of feature set `X`.
            self.model.evaluate(np.zeros((n, d)), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of observation set `Y`.
            self.model.evaluate(X, None, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # ndarray instead of observation set `Y`.
            self.model.evaluate(X, np.zeros((n, 1)), params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible feature set.
            self.model.evaluate(random_matrix((d, n)), Y, params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible observation set.
            self.model.evaluate(X, random_matrix((n + 1, 1)), params=params)

        with self.assertRaises(_IncompleteModelError):
            # None instead of model parameters `params`.
            self.model.evaluate(X, Y, params=None)

        with self.assertRaises(_InvalidModelParametersError):
            # List instead of model parameters tuple `params`.
            self.model.evaluate(X, Y, params=list(params))

        with self.assertRaises(_InvalidModelParametersError):
            # List of ndarray instead of np.matrix tuple `params`.
            self.model.evaluate(X, Y,
                                params=compose(tuple,
                                               map)(np.zeros, self.shapes))

        with self.assertRaises(TypeError):
            # None instead of string `loss_fn`.
            self.model.evaluate(X, Y, params=params, loss_fn=None)

        with self.assertRaises(TypeError):
            # Integer instead of string `loss_fn`.
            self.model.evaluate(X, Y, params=params, loss_fn=123)

        with self.assertRaises(AttributeError):
            # Non-existent loss function.
            self.model.evaluate(X, Y, params=params, loss_fn="non-existent")

    def test_invalid_args_model_gradient(self):
        """`Model.gradient`: Argument Validator.

        Tests the behavior of `gradient` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = random_matrix((n, 1))
        """np.matrix: Random-valued observation set."""
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.gradient()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.gradient(X, Y, Y, params=params)

        with self.assertRaises(_IncompleteModelError):
            # Params not set.
            self.model.gradient(X, Y)

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            self.model.gradient(X, Y, params=params, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of feature set `X`.
            self.model.gradient(None, Y, params=params)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of feature set `X`.
            self.model.gradient(np.zeros((n, d)), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of observation set `Y`.
            self.model.gradient(X, None, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # ndarray instead of observation set `Y`.
            self.model.gradient(X, np.zeros((n, 1)), params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible feature set.
            self.model.gradient(random_matrix((d, n)), Y, params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible observation set.
            self.model.gradient(X, random_matrix((n + 1, 1)), params=params)

        with self.assertRaises(_IncompleteModelError):
            # None instead of model parameters `params`.
            self.model.gradient(X, Y, params=None)

        with self.assertRaises(_InvalidModelParametersError):
            # List instead of model parameters tuple `params`.
            self.model.gradient(X, Y, params=list(params))

        with self.assertRaises(_InvalidModelParametersError):
            # List of ndarray instead of np.matrix tuple `params`.
            self.model.gradient(X, Y,
                                params=compose(tuple,
                                               map)(np.zeros, self.shapes))

    def test_invalid_args_model_numerical_gradient(self):
        """`Model.numerical_gradient`: Argument Validator.

        Tests the behavior of `numerical_gradient` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        Y = random_matrix((n, 1))
        """np.matrix: Random-valued observation set."""

        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.numerical_gradient()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.numerical_gradient(X, Y, Y, params=params)

        with self.assertRaises(_IncompleteModelError):
            # Params not set.
            self.model.numerical_gradient(X, Y)

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            self.model.numerical_gradient(X, Y, params=params, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of feature set `X`.
            self.model.numerical_gradient(None, Y, params=params)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of feature set `X`.
            self.model.numerical_gradient(np.zeros((n, d)), Y, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of observation set `Y`.
            self.model.numerical_gradient(X, None, params=params)

        with self.assertRaises(_InvalidObservationSetError):
            # ndarray instead of observation set `Y`.
            self.model.numerical_gradient(X, np.zeros((n, 1)), params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible feature set.
            self.model.numerical_gradient(random_matrix((d, n)), Y,
                                          params=params)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible observation set.
            self.model.numerical_gradient(X, random_matrix((n + 1, 1)),
                                          params=params)

        with self.assertRaises(_IncompleteModelError):
            # None instead of model parameters `params`.
            self.model.numerical_gradient(X, Y, params=None)

        with self.assertRaises(_InvalidModelParametersError):
            # List instead of model parameters tuple `params`.
            self.model.numerical_gradient(X, Y, params=list(params))

        with self.assertRaises(_InvalidModelParametersError):
            # List of ndarray instead of np.matrix tuple `params`.
            self.model.numerical_gradient(X, Y,
                                          params=compose(tuple,
                                                         map)(np.zeros,
                                                              self.shapes))

    def test_invalid_args_model_predict(self):
        """`Model.predict`: Argument Validator.

        Tests the behavior of `predict` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.predict()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.predict(X, X, params=params)

        with self.assertRaises(_IncompleteModelError):
            # Params not set.
            self.model.predict(X)

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            self.model.predict(X, params=params, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of feature set `X`.
            self.model.predict(None, params=params)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of feature set `X`.
            self.model.predict(np.zeros((n, d)), params=params)

        with self.assertRaises(_InvalidModelParametersError):
            # Incompatible observation set.
            self.model.predict(X, params=(random_matrix((n + 1, 1)),))

        with self.assertRaises(_IncompleteModelError):
            # None instead of model parameters `params`.
            self.model.predict(X, params=None)

        with self.assertRaises(_InvalidModelParametersError):
            # List instead of model parameters tuple `params`.
            self.model.predict(X, params=list(params))

        with self.assertRaises(_InvalidModelParametersError):
            # List of ndarray instead of np.matrix tuple `params`.
            self.model.predict(X, params=compose(tuple,
                                                 map)(np.zeros, self.shapes))

    def test_invalid_args_model_regularization(self):
        """`Model.regularization`: Argument Validator.

        Tests the behavior of `regularization` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        params = compose(tuple, map)(random_matrix, self.shapes)
        """tuple of np.matrix: Random-valued parameters."""

        with self.assertRaises(_IncompleteModelError):
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

    def test_invalid_args_model_train(self):
        """`Model.train`: Argument Validator.

        Tests the behavior of `train` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        Y = random_matrix((n, 1))
        """np.matrix: Random-valued observation set."""

        with self.assertRaises(TypeError):
            # No arguments.
            self.model.train()

        with self.assertRaises(TypeError):
            # Too many arguments.
            self.model.train(X, Y, Y)

        with self.assertRaises(TypeError):
            # Kwarg.
            self.model.train(X, Y, key="value")

        with self.assertRaises(_InvalidFeatureSetError):
            # `None` instead of feature set `X`.
            self.model.train(None, Y)

        with self.assertRaises(_InvalidFeatureSetError):
            # ndarray instead of feature set `X`.
            self.model.train(np.zeros((n, d)), Y)

        with self.assertRaises(_InvalidObservationSetError):
            # `None` instead of observation set `Y`.
            self.model.train(X, None)

        with self.assertRaises(_InvalidObservationSetError):
            # ndarray instead of observation set `Y`.
            self.model.train(X, np.zeros((n, 1)))

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible feature set.
            self.model.train(random_matrix((d, n)), Y)

        with self.assertRaises(_IncompatibleDataSetsError):
            # Incompatible observation set.
            self.model.train(X, random_matrix((n + 1, 1)))

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

    def test_random_model_evaluate(self):
        """`Model.evaluate`: Randomized Validator.

        Tests the behavior of `evaluate` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `evaluate` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""

            Y = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            random_params = compose(tuple, map)(random_matrix, self.shapes)
            """tuple of np.matrix: Random-valued parameters."""

            # First, test `params` as a method argument.
            result1 = self.model.evaluate(X, Y, params=random_params)
            """float: Test input 1."""

            # Gradients should be a tuple.
            self.assertEqual(type(result1), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result1), 2)

            err1, Y_hat1 = result1
            """(float, np.matrix): Evaluation error and predicted observations
            of test 1."""

            # Evaluation error should be a float.
            self.assertEqual(type(err1), np.float64)

            # Prediction set should be a matrix.
            self.assertEqual(type(Y_hat1), np.matrix)

            # Model parameters should not be set at this point.
            self.assertIsNone(self.model.params)

            # Finally, test `params` as attribute.
            self.model.params = random_params

            result2 = self.model.evaluate(X, Y)
            """float: Test input 2."""

            # Gradients should be a tuple.
            self.assertEqual(type(result2), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result2), 2)

            err2, Y_hat2 = result2
            """(float, np.matrix): Evaluation error and predicted observations
            of test 2."""

            # Evaluation error should be a float.
            self.assertEqual(type(err2), np.float64)

            # Prediction set should be a matrix.
            self.assertEqual(type(Y_hat2), np.matrix)

            # Model parameters should be set at this point.
            self.assertIsNotNone(self.model.params)

            # Evaluation errors should match.
            self.assertEqual(err1, err2)

            # Norms of test inputs should match.
            self.assertEqual(np.linalg.norm(Y_hat1), np.linalg.norm(Y_hat2))

            r = self.model.regularization()
            """float: L2 parameter regularization."""

            (err_no_reg,
             Y_hat_no_reg) = self.model.evaluate(X, Y, regularize=False)
            """(float, np.matrix): Evaluation error and predicted observations
            of test 2."""

            # Evaluation with no regularization should comply with the following
            # equation.
            self.assertEqual(err1, err_no_reg * self.data_shape[0] + r)

            # Predicted observations should still be identical, though.
            self.assertEqual(np.linalg.norm(Y_hat1),
                             np.linalg.norm(Y_hat_no_reg))

            del self.model.params

    def test_random_model_gradient(self):
        """`Model.gradient`: Randomized Validator.

        Tests the behavior of `gradient` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `gradient` needs debugging.

        """
        (norms1,
         norms2) = map(np.matrix, self.model.gradient_checker(self.n_tests))
        """(np.matrix, np.matrix): """

        # The norms of gradients generated numerically should match those
        # generated analytically.
        self.assertLessEqual(compose(np.square,
                                     np.linalg.norm,
                                     np.transpose,
                                     np.subtract)(norms1, norms2),
                             1e-2 * self.n_tests)

        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""

            Y = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            params = compose(tuple, map)(random_matrix, self.shapes)
            """tuple of np.matrix: Random-valued parameters."""

            # First, test `params` as a method argument.
            result1 = self.model.gradient(X, Y, params=params)
            """float: Test input 1."""

            # Gradients should be a tuple.
            self.assertEqual(type(result1), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result1), len(self.shapes))

            # All gradients should matrices.
            for g in result1:
                self.assertEqual(type(g), np.matrix)

            # Model parameters should not be set at this point.
            self.assertIsNone(self.model.params)

            # Finally, test `params` as attribute.
            self.model.params = params

            result2 = self.model.gradient(X, Y)
            """float: Test input 2."""

            # Gradients should be a tuple.
            self.assertEqual(type(result2), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result2), len(self.shapes))

            # All gradients should matrices.
            for g in result2:
                self.assertEqual(type(g), np.matrix)

            # Model parameters should be set at this point.
            self.assertIsNotNone(self.model.params)

            norm1 = compose(sum, map)(np.linalg.norm, result1)
            """float: Sum of `result1`'s gradient norms."""
            norm2 = compose(sum, map)(np.linalg.norm, result2)
            """float: Sum of `result2`'s gradient norms."""

            # Norms of test inputs should match.
            self.assertEqual(norm1, norm2)

            del self.model.params

    def test_random_model_numerical_gradient(self):
        """`Model.numerical_gradient`: Randomized Validator.

        Tests the behavior of `numerical_gradient` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `numerical_gradient` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""

            Y = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            random_params = compose(tuple, map)(random_matrix, self.shapes)
            """tuple of np.matrix: Random-valued parameters."""

            # First, test `params` as a method argument.
            result1 = self.model.numerical_gradient(X, Y, params=random_params)
            """float: Test input 1."""

            # Gradients should be a tuple.
            self.assertEqual(type(result1), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result1), len(self.shapes))

            # All gradients should matrices.
            for g in result1:
                self.assertEqual(type(g), np.matrix)

            # Model parameters should not be set at this point.
            self.assertIsNone(self.model.params)

            # Finally, test `params` as attribute.
            self.model.params = random_params

            result2 = self.model.numerical_gradient(X, Y)
            """float: Test input 2."""

            # Gradients should be a tuple.
            self.assertEqual(type(result2), tuple)

            # All params should have a gradient.
            self.assertEqual(len(result2), len(self.shapes))

            # All gradients should matrices.
            for g in result2:
                self.assertEqual(type(g), np.matrix)

            # Model parameters should be set at this point.
            self.assertIsNotNone(self.model.params)

            norm1 = compose(sum, map)(np.linalg.norm, result1)
            """float: Sum of `result1`'s gradient norms."""
            norm2 = compose(sum, map)(np.linalg.norm, result2)
            """float: Sum of `result2`'s gradient norms."""

            # Norms of test inputs should match.
            self.assertEqual(norm1, norm2)

            del self.model.params

    def test_random_model_predict(self):
        """`Model.predict`: Randomized Validator.

        Tests the behavior of `predict` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `predict` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""

            params = compose(tuple, map)(random_matrix, self.shapes)
            """tuple of np.matrix: Random-valued parameters."""

            # First, test `params` as a method argument.
            Y_hat1 = self.model.predict(X, params=params)
            """np.matrix: Test input 1."""

            # Gradients should be a tuple.
            self.assertEqual(type(Y_hat1), np.matrix)

            # All params should have a gradient.
            self.assertEqual(Y_hat1.shape, (X.shape[0], 1))

            # Model parameters should not be set at this point.
            self.assertIsNone(self.model.params)

            # Finally, test `params` as attribute.
            self.model.params = params

            Y_hat2 = self.model.predict(X)
            """np.matrix: Test input 2."""

            # Gradients should be a tuple.
            self.assertEqual(type(Y_hat2), np.matrix)

            # All params should have a gradient.
            self.assertEqual(Y_hat2.shape, (X.shape[0], 1))

            # Model parameters should be set at this point.
            self.assertIsNotNone(self.model.params)

            # Norms of test inputs should match.
            self.assertEqual(np.linalg.norm(Y_hat1), np.linalg.norm(Y_hat2))

            del self.model.params

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

    def test_random_model_train(self):
        """`Model.train`: Randomized Validator.

        Tests the behavior of `train` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `train` needs debugging.

        """
        for i in range(0, self.n_tests):
            X = random_matrix(self.data_shape)
            """np.matrix: Random-valued feature set."""
            Y = random_matrix((self.data_shape[0], 1))
            """np.matrix: Random-valued observation set."""

            # Model parameters should be uninitialized at this point.
            self.assertIsNone(self.model.params)

            self.model.init_params(X)

            # Model parameters should be set at this point.
            self.assertIsNotNone(self.model.params)

            err = self.model.evaluate(X, Y)[0]
            """float: Evaluation error prior to training."""
            train_err = self.model.train(X, Y)
            """float: Test input."""

            # Evaluation error should be number.
            self.assertEqual(type(train_err), np.float64)

            # Evaluation error prior to training should larger than after
            # training.
            self.assertLess(train_err, err)

            del self.model.params
