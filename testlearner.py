"""Learner Testing Module.

"""
import numpy as _np
import math as _math
import unittest as _unittest
import re as _re

from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidModelParametersError as _InvalidModelParametersError
from common.test_cases.module_test_case import ModuleTestCase as _ModuleTest
from learner import Learner
from utils.linalg import random_matrix as  _random_matrix
from utils.general import compose as _compose


class _Test(_ModuleTest):
    """Learner Unit Tester.

    Runs tests for all properties and methods of the `Learner`class:
        - `_update_rule`

    Attributes:
        cutoff_zero (float): The largest value treated as zero in all equality
            tests.
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the linear model module.
        learner (Learner): Instance of `Learner`.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.
        shapes (:obj:`tuple` of np.matrix): Matrix dimensions for all parameters
            in all models.

    """
    def setUp(self):
        """Learner Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.cutoff_zero = 1e-2
        self.data_shape = 100, 20
        self.label = '`learner.Learner`'
        self.learner = Learner()
        self.n_tests = 50
        self.name = __name__
        self.shapes = {}

        p_to_shape = lambda p: p.shape
        """callable: Maps parameters to their matrix dimensions."""

        for name, ModelWrapper in self.learner:
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
        """`Learner._ModelWrapper.model`: Edge Case Validator.

        Tests the behavior of `model` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(_InvalidModelParametersError):
            # Empty parameters.
            self.learner.Linear.model = dict(params=(_np.matrix([[]]),))

        with self.assertRaises(_InvalidFeatureSetError):
            # Empty feature set.
            self.learner.Linear.model = dict(X=_np.matrix([[]]))

    def test_invalid_args_model_wrapper_model(self):
        """`Learner._ModelWrapper.model`: Argument Validator.

        Tests the behavior of `model` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        for name, ModelWrapper in self.learner:
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

    def test_random_model_wrapper_model(self):
        """`Learner._ModelWrapper.model`: Randomized Validator.

        Tests the behavior of `model` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `model` needs debugging.

        """
        shape_regex = _re.compile(r"^\([0-9]+, [0-9]+\)$")
        """SRE_Pattern: Identifies stringified shapes."""

        for i in range(0, self.n_tests):
            X = _random_matrix(self.data_shape)
            """np.matrix: Random-valued matrix."""

            for name, ModelWrapper in self.learner:
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

    def _params_set(self, name, s):
        """

        Runs tests to check that the given parameter string matches
        a stringified parameter shape.

        Args:
            s (str): Stringified parameter.

        Raises:
            AssertionError: If `s` does not indicate that the parameter has been
                properly set.

        """
        self.assertIn(s, map(str, self.shapes[name]))

    def _params_unset(self, name, s):
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
            value_test(name, value)


Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: Linear model testing suite."""
