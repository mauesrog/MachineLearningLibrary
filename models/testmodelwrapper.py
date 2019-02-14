"""Model Wrapper Testing Module.

Attributes:
    Test (TestSuite): ModelWrapper testing suite.
    _active_models (:obj:`Model`): All available learning models.

"""
import numpy as _np
from unittest import TestLoader as _TestLoader
import re as _re

from common.exceptions import IncompleteModelError as _IncompleteModelError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidModelParametersError as _InvalidModelParametersError, \
                              InvalidObservationSetError as _InvalidObservationSetError
from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from models.linear import LinearModel as _LinearModel
from modelwrapper import ModelWrapper as ModelWrapperClass
from utils.linalg import random_matrix as  _random_matrix
from utils.general import compose as _compose

_active_models = dict(Linear=_LinearModel)


class _Test(_ModuleTestCase):
    """Model Wrapper Unit Tester.

    Runs tests for all properties and methods of the `ModelWrapper` class:
        - `model`: getter and setter
        - `evaluate`
        - `predict`

    Attributes:
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the linear model module.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.
        shapes (:obj:`tuple` of np.matrix): Matrix dimensions for all parameters
            in all models.
        wrappers (:obj:`ModelWrapper`): Contains all available wrapped models.

    """
    def setUp(self):
        """Model Wrapper Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.data_shape = 100, 20
        self.label = '`learner.Learner`'
        self.n_tests = 50
        self.name = __name__
        self.shapes = {}
        self.wrappers = {
            n: ModelWrapperClass(m) for n, m in _active_models.iteritems()
        }

        p_to_shape = lambda p: p.shape
        """callable: Maps parameters to their matrix dimensions."""

        for name, ModelWrapper in self.wrappers.iteritems():
            ModelWrapper.model = dict(X=_random_matrix(self.data_shape))

            self.shapes[name] = _compose(tuple,
                                         map)(p_to_shape,
                                              ModelWrapper._model.params)

            # Model string should indicate that all parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_set)

            del ModelWrapper._model.params

            # Model string should indicate unset parameters at this point.
            self._validate_model_getter(ModelWrapper, name, self._params_unset)

    def test_edge_cases_model_wrapper_model(self):
        """`modelwrapper.ModelWrapper.model`: Edge Case Validator.

        Tests the behavior of `ModelWrapper.model` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        for name, ModelWrapper in self.wrappers.iteritems():
            with self.assertRaises(_InvalidModelParametersError):
                # Empty parameters.
                ModelWrapper.model = dict(params=(_np.matrix([[]]),))

            with self.assertRaises(_InvalidFeatureSetError):
                # Empty feature set.
                ModelWrapper.model = dict(X=_np.matrix([[]]))

    def test_edge_cases_model_wrapper_evaluate(self):
        """`modelwrapper.ModelWrapper.evaluate`: Edge Case Validator.

        Tests the behavior of `ModelWrapper.evaluate` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix(self.data_shape)
        """np.matrix: Random-valued observation set."""

        for name, ModelWrapper in self.wrappers.iteritems():
            # Model string should indicate that no parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_unset)

            ModelWrapper.model = dict(X=X)

            # Model string should indicate that all parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_set)

            with self.assertRaises(_InvalidFeatureSetError):
                # Empty feature set.
                ModelWrapper.evaluate(_np.matrix([[]]), Y)

            with self.assertRaises(_InvalidObservationSetError):
                # Empty observation set.
                ModelWrapper.evaluate(X, _np.matrix([[]]))

    def test_edge_cases_model_wrapper_predict(self):
        """`modelwrapper.ModelWrapper.predict`: Edge Case Validator.

        Tests the behavior of `ModelWrapper.predict` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        for name, ModelWrapper in self.wrappers.iteritems():
            # Model string should indicate that no parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_unset)

            ModelWrapper.model = dict(X=X)

            # Model string should indicate that all parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_set)

            with self.assertRaises(_InvalidFeatureSetError):
                # Empty feature set.
                ModelWrapper.predict(_np.matrix([[]]))

    def test_invalid_args_model_wrapper_model(self):
        """`modelwrapper.ModelWrapper.model`: Argument Validator.

        Tests the behavior of `ModelWrapper.model` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        for name, ModelWrapper in self.wrappers.iteritems():
            params = _compose(tuple, map)(_random_matrix, self.shapes[name])
            """tuple of np.matrix: Random-valued parameters."""

            with self.assertRaises(AttributeError):
                # Too many parameters.
                ModelWrapper.model = dict(params=params), dict(params=params)

            with self.assertRaises(AttributeError):
                # No parameters set.
                ModelWrapper.model = {}

            with self.assertRaises(AttributeError):
                # Both parameters set.
                ModelWrapper.model = dict(params=params,
                                          X=_random_matrix(self.data_shape))

            with self.assertRaises(_InvalidModelParametersError):
                # `None` instead of parameters.
                ModelWrapper.model = dict(params=None)

            with self.assertRaises(_InvalidModelParametersError):
                # List instead of parameter tuple.
                ModelWrapper.model = dict(params=list(params))

            with self.assertRaises(_InvalidModelParametersError):
                ndarray_parms = dict(params=_compose(tuple,
                                                     map)(_np.zeros,
                                                          self.shapes[name]))

                # Tuple of ndarray instead of matrix tuple.
                ModelWrapper.model = ndarray_parms

            with self.assertRaises(_InvalidFeatureSetError):
                # `None` instead of feature set `X`.
                ModelWrapper.model = dict(X=None)

            with self.assertRaises(_InvalidFeatureSetError):
                # ndarray instead of matrix `X`.
                ModelWrapper.model = dict(X=_np.zeros(self.data_shape))

    def test_invalid_args_model_wrapper_evaluate(self):
        """`modelwrapper.ModelWrapper.evaluate`: Argument Validator.

        Tests the behavior of `ModelWrapper.evaluate` with invalid argument
        counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued feature set."""

        for name, ModelWrapper in self.wrappers.iteritems():
            # Model string should indicate that no parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_unset)

            with self.assertRaises(_IncompleteModelError):
                # Unset parameters.
                ModelWrapper.evaluate(X, Y)

            ModelWrapper.model = dict(X=X)

            # Model string should indicate that all parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_set)

            with self.assertRaises(TypeError):
                # No arguments.
                ModelWrapper.evaluate()

            with self.assertRaises(TypeError):
                # Too many arguments.
                ModelWrapper.evaluate(X, Y, X)

            with self.assertRaises(TypeError):
                # Invalid kwarg.
                ModelWrapper.evaluate(X, Y, key="value")

            with self.assertRaises(_InvalidFeatureSetError):
                # `None` instead of feature set `X`.
                ModelWrapper.evaluate(None, Y)

            with self.assertRaises(_InvalidFeatureSetError):
                # ndarray instead of feature set `X`.
                ModelWrapper.evaluate(_np.zeros(self.data_shape), Y)

            with self.assertRaises(_InvalidObservationSetError):
                # `None` instead of observation set `Y`.
                ModelWrapper.evaluate(X, None)

            with self.assertRaises(_InvalidObservationSetError):
                # ndarray instead of observation set `Y`.
                ModelWrapper.evaluate(X, _np.zeros((self.data_shape[0], 1)))

    def test_invalid_args_model_wrapper_predict(self):
        """`modelwrapper.ModelWrapper.predict`: Argument Validator.

        Tests the behavior of `ModelWrapper.predict` with invalid argument
        counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        n, d = self.data_shape
        """(int, int): Number of data points and number of features."""

        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""

        for name, ModelWrapper in self.wrappers.iteritems():
            # Model string should indicate that no parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_unset)

            with self.assertRaises(_IncompleteModelError):
                # Unset parameters.
                ModelWrapper.predict(X)

            ModelWrapper.model = dict(X=X)

            # Model string should indicate that all parameters are set at this
            # point.
            self._validate_model_getter(ModelWrapper, name, self._params_set)

            with self.assertRaises(TypeError):
                # No arguments.
                ModelWrapper.predict()

            with self.assertRaises(TypeError):
                # Too many arguments.
                ModelWrapper.predict(X, X)

            with self.assertRaises(TypeError):
                # Invalid kwarg.
                ModelWrapper.predict(X, key="value")

            with self.assertRaises(_InvalidFeatureSetError):
                # `None` instead of feature set `X`.
                ModelWrapper.predict(None)

            with self.assertRaises(_InvalidFeatureSetError):
                # ndarray instead of feature set `X`.
                ModelWrapper.predict(_np.zeros((n, d)))

    def test_random_model_wrapper_model(self):
        """`modelwrapper.ModelWrapper.model`: Randomized Validator.

        Tests the behavior of `ModelWrapper.model` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `ModelWrapper.model` needs debugging.

        """
        shape_regex = _re.compile(r"^\([0-9]+, [0-9]+\)$")
        """SRE_Pattern: Identifies stringified shapes."""

        for i in range(self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""

            for name, ModelWrapper in self.wrappers.iteritems():
                # Model string should indicate unset parameters at this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_unset)

                params = _compose(tuple, map)(_random_matrix, self.shapes[name])
                """tuple of np.matrix: Random-valued parameters."""

                ModelWrapper.model = dict(params=params)

                # Model string should indicate that all parameters are set at
                # this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_set)

                curr_params = ModelWrapper._model.params
                """tuple of np.matrix: Currently set model parameters."""

                # Number of parameters in output should match input number.
                self.assertEqual(len(curr_params), len(params))

                # Matrix norm sums should match for input and output.
                self.assertEqual(*map(_np.linalg.norm, [curr_params, params]))

                del ModelWrapper._model.params

                # Model string should indicate unset parameters again.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_unset)

                ModelWrapper.model = dict(X=X)

                # Model string should indicate that all parameters are set once
                # more.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_set)
                del ModelWrapper._model.params

    def test_random_model_wrapper_evaluate(self):
        """`modelwrapper.ModelWrapper.evaluate`: Randomized Validator.

        Tests the behavior of `ModelWrapper.evaluate` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `ModelWrapper.evaluate` needs debugging.

        """
        for i in range(self.n_tests):
            for name, ModelWrapper in self.wrappers.iteritems():
                X = _random_matrix(self.data_shape)
                """np.matrix: Random-valued feature set."""
                Y = _random_matrix((self.data_shape[0], 1))
                """np.matrix: Random-valued observation set."""

                # Model string should indicate that no parameters are set at
                # this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_unset)

                ModelWrapper.model = dict(X=X)

                err = ModelWrapper.evaluate(X, Y)
                """np.matrix: Test input 1."""

                # Loss should be a float.
                self.assertEqual(type(err), _np.float64)

                # Loss should be a positive integer.
                self.assertGreaterEqual(err, 0.0)

                # Model string should indicate that all parameters are set at
                # this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_set)

                del ModelWrapper._model.params

    def test_random_model_wrapper_predict(self):
        """`modelwrapper.ModelWrapper.predict`: Randomized Validator.

        Tests the behavior of `ModelWrapper.predict` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `ModelWrapper.predict` needs debugging.

        """
        for i in range(self.n_tests):
            for name, ModelWrapper in self.wrappers.iteritems():
                X = _random_matrix(self.data_shape)
                """np.matrix: Random-valued feature set."""

                # Model string should indicate that no parameters are set at
                # this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_unset)

                ModelWrapper.model = dict(X=X)

                # First, test `params` as a method argument.
                Y_hat = ModelWrapper.predict(X)
                """np.matrix: Test input 1."""

                # Gradients should be a tuple.
                self.assertIsInstance(Y_hat, _np.matrix)

                # All params should have a gradient.
                self.assertEqual(Y_hat.shape, (X.shape[0], 1))

                # Model string should indicate that all parameters are set at
                # this point.
                self._validate_model_getter(ModelWrapper, name,
                                            self._params_set)

                del ModelWrapper._model.params

    def _params_set(self, s):
        """

        Runs tests to check that the given parameter string matches
        a stringified parameter shape.

        Args:
            s (str): Stringified parameter.

        Raises:
            AssertionError: If `s` does not indicate that the parameter has been
                properly set.

        """
        self.assertRegexpMatches(s, r"^\([0-9]+, [0-9]+\)$")

    def _params_unset(self, s):
        """

        Runs tests to check that the given parameter string reflects that it is
        not currently set.

        Args:
            s (str): Stringified parameter.

        Raises:
            AssertionError: If `s` indicates that the parameter has been
                properly set.

        """
        self.assertEqual(s, "not set")

    def _validate_model_getter(self, ModelWrapper, name, value_test):
        """`_ModelWrapper.model`: Getter Validator.

        Args:
            ModelWrapper (_ModelWrapper): Instance getting validated.
            name (str): Key for current `_ModelWrapper`.
            value_test (callable): Runs tests that validate getter output.

        Raises:
            AssertionError: If the `model` getter needs debugging.

        """
        param_extractor = lambda s: _compose(tuple,
                                             s.rstrip(",").split)("`: ")
        """callable: Separates stringified key/parameters pairs."""

        param_regex = _re.compile("^%sModel:" % name)
        """SRE_Pattern: Identifies the current `_ModelWrapper`'s full
        name."""
        param_labels = map(param_extractor,
                           filter(lambda s: len(s),
                                  param_regex.sub("",
                                                  ModelWrapper.model).split(" `")))
        """list of (str, str): String representations of parameter names
        and values."""

        # Only registered parameters should be printed out.
        self.assertEqual(len(ModelWrapper._model._param_names),
                         len(param_labels))

        for p_name, value in param_labels:
            # Only registered parameters should be printed out.
            self.assertIn("_%s" % p_name, ModelWrapper._model._param_names)

            # All parameters should be uninitialized at this point.
            value_test(value)

Test = _TestLoader().loadTestsFromTestCase(_Test)
