"""Learner Testing Module.

Attributes:
    Test (TestSuite): Model wrapper testing suite.

Todo:
    - Finish unit tests for `Learner`'s private methods.

"""
import numpy as _np
from unittest import TestLoader as _TestLoader

from config import data_examples as _examples
from common.exceptions import InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidObservationSetError as _InvalidObservationSetError
from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from learner import Learner
from utils.linalg import random_matrix as  _random_matrix
from utils.general import compose as _compose
from utils.stats import partition_data as _partition_data, \
                        reduce_dimensions as _reduce_dimensions


class _Test(_ModuleTestCase):
    """Learner Unit Tester.

    Runs tests for all properties and methods of the `Learner` class:

    And `ModelWrapper` class:
        - `train`


    Attributes:
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

            self.shapes[name] = _compose(tuple, map)(p_to_shape,
                                                     ModelWrapper._model.params)

            # Model string should indicate that all parameters are set at this
            # point.
            self.assertIsNotNone(ModelWrapper._model.params)

            del ModelWrapper._model.params

            # Model string should indicate unset parameters at this point.
            self.assertIsNone(ModelWrapper._model.params)

    def test_edge_cases_model_wrapper_train(self):
        """`modelwrapper.ModelWrapper.train`: Edge Case Validator.

        Tests the behavior of `ModelWrapper.train` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        for name, ModelWrapper in self.learner:
            with self.assertRaises(_InvalidFeatureSetError):
                # Empty feature set.
                ModelWrapper.train(_np.matrix([[]]), Y)

            with self.assertRaises(_InvalidObservationSetError):
                # Empty observation set.
                ModelWrapper.train(X, _np.matrix([[]]))

            with self.assertRaises(ValueError):
                # Incompatible `k` value.
                ModelWrapper.train(X, Y, exact=True, k=self.data_shape[0])

    def test_example_model_wrapper_train(self):
        """`modelwrapper.ModelWrapper.train`: Example Validator.

        Tests the behavior of `ModelWrapper.train` by feeding it real-life
        datasets.

        Raises:
            AssertionError: If `ModelWrapper.train` needs debugging.

        """
        for name, ModelWrapper in self.learner:
            for type, loader in _examples[name].iteritems():
                if type == "classification":
                    break

                X, Y, desc, feature_names, target, kwargs = loader()[:-1]
                """tuple: Example dataset."""

                train_X, train_Y, test_X, test_Y = _partition_data(X, Y, 0.6)
                """(np.matrix, np.matrix, np.matrix, np.matrix): Training and
                testing feature and observation sets."""

                ModelWrapper.model = dict(X=train_X)

                init_err = ModelWrapper.evaluate(test_X, test_Y)
                """float: Testing error before training."""

                a_training_err = ModelWrapper.train(train_X, train_Y,
                                                    exact=True)
                """float: Analytical training error."""
                a_err = ModelWrapper.evaluate(test_X, test_Y)
                """float: Testing error after analytical training."""

                v = self.learner.visualize(desc, (3, 5))
                v.plot_features(test_X, test_Y, None, ModelWrapper,
                                feature_names, target)
                v.show()
                v.close()

                n_training_err = ModelWrapper.train(train_X, train_Y, **kwargs)
                """float: Numerical training error."""
                n_err = ModelWrapper.evaluate(test_X, test_Y)
                """float: Testing error after numerical training."""


                # Analytical training should provide a global minimum.
                self.assertLess(a_err, init_err)
                self.assertLess(a_err, n_err)
                self.assertLess(a_training_err, n_training_err)

                # Numerical training should provide a better guess than the
                # initial one.
                self.assertLess(n_err, init_err)

    def test_invalid_args_model_wrapper_train(self):
        """`modelwrapper.ModelWrapper.train`: Argument Validator.

        Tests the behavior of `ModelWrapper.train` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        for name, ModelWrapper in self.learner:
            with self.assertRaises(TypeError):
                # No arguments.
                ModelWrapper.train()

            with self.assertRaises(TypeError):
                # Too many arguments.
                ModelWrapper.train(X, Y, X)

            with self.assertRaises(TypeError):
                # Invalid kwarg.
                ModelWrapper.train(X, Y, key="value")

            with self.assertRaises(_InvalidFeatureSetError):
                # `None` instead of feature set `X`.
                ModelWrapper.train(None, Y)

            with self.assertRaises(_InvalidFeatureSetError):
                # ndarray instead of feature set `X`.
                ModelWrapper.train(_np.zeros(self.data_shape), Y)

            with self.assertRaises(_InvalidObservationSetError):
                # `None` instead of observation set `Y`.
                ModelWrapper.train(X, None)

            with self.assertRaises(_InvalidObservationSetError):
                # ndarray instead of observation set `Y`.
                ModelWrapper.train(X, _np.zeros((self.data_shape[0], 1)))

            with self.assertRaises(TypeError):
                # None instead of int `k`.
                ModelWrapper.train(X, Y, k=None)

            with self.assertRaises(TypeError):
                # Float instead of int `k`.
                ModelWrapper.train(X, Y, k=0.5)

            with self.assertRaises(ValueError):
                # Negative integer instead of positive integer `k`.
                ModelWrapper.train(X, Y, k=-10)

            with self.assertRaises(ValueError):
                # Zero of positive integer `k`.
                print ModelWrapper.train(X, Y, k=0)

    def test_random_model_wrapper_train(self):
        """`modelwrapper.ModelWrapper.train`: Randomized Validator.

        Tests the behavior of `ModelWrapper.train` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `ModelWrapper.train` needs debugging.

        """
        n = self.data_shape[0]
        """int: Total number of data points."""

        for i in range(self.n_tests):
            for name, ModelWrapper in self.learner:
                for percent in range(10, 100, 18):
                    f = percent / 100.0
                    """float: Training to testing ratio."""

                    min_k = min(10, _compose(int, _np.floor)(f / 2.0 * n))
                    """int: Minimum number of data points per batch."""
                    max_k = _compose(int, _np.floor)(f * n)
                    """int: Minimum number of data points per batch."""

                    for k in range(min_k, max_k,
                                   _compose(int,
                                            _np.floor)((max_k - min_k) / 5)):
                        X = _random_matrix(self.data_shape)
                        """np.matrix: Random-valued feature set."""
                        Y = _random_matrix((n, 1))
                        """np.matrix: Random-valued observation set."""

                        train_X, train_Y, test_X, test_Y = _partition_data(X, Y,
                                                                           f)
                        """(np.matrix, np.matrix, np.matrix, np.matrix): Train-
                        ing and testing feature and observation sets."""


                        ModelWrapper.model = dict(X=train_X)

                        init_err = ModelWrapper.evaluate(test_X, test_Y)
                        """float: Loss before training."""

                        ModelWrapper.train(train_X, train_Y, exact=True, k=k)

                        a_err = ModelWrapper.evaluate(test_X, test_Y)
                        """float: Loss after analytical training."""

                        # Analytical error should be a float.
                        self.assertIsInstance(a_err, float)

                        if f >= 0.5:
                            # Analytical training should provide a global
                            # minimum.
                            self.assertLess(a_err, init_err)

Test = _TestLoader().loadTestsFromTestCase(_Test)
