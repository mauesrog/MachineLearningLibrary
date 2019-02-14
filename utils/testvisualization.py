"""Visualization Testing Module.

Attributes:
    Test (TestSuite): Visualization testing suite.

"""
import unittest as _unittest
import numpy as _np
from matplotlib.lines import Line2D as _Line2D

from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from common.exceptions import IncompatibleDataSetsError as _IncompatibleDataSetsError, \
                              InvalidFeatureSetError as _InvalidFeatureSetError, \
                              InvalidObservationSetError as _InvalidObservationSetError
from models.linear import LinearModel as _LinearModel
from models.modelwrapper import ModelWrapper as _ModelWrapperClass
from linalg import random_matrix as _random_matrix
from general import appendargs as _appendargs, compose as _compose, \
                    prependargs as _prependargs
from visualization import Visualization

_active_models = dict(Linear=_LinearModel)


class _Test(_ModuleTestCase):
    """Visualization Unit Tester.

    Runs tests for `Visualization` methods:
        - `init`
        - `_generate_color`
        - `_empty_subplot`
        - `_plot_feature`

    Attributes:
        data_shape ((int, int)): Dimensions for all auto-generated data sets.
        label (str): Identifier for super class to generate custome test
            docstrings according to the general model utilities module. See base
            class.
        max_suplots (int): Maximum number of suplots in randomly generated
            layouts.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.
        shapes (:obj:`tuple` of np.matrix): Matrix dimensions for all parameters
            in all models.
        wrappers (:obj:`ModelWrapper`): Contains all available wrapped models.

    """
    def setUp(self):
        """General Utilities Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.data_shape = 100, 20
        self.label = '`visualization`'
        self.max_suplots = 20
        self.n_tests = 20
        self.name = __name__
        self.shapes = {}
        self.wrappers = {
            n: _ModelWrapperClass(m) for n, m in _active_models.iteritems()
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
            self.assertIsNotNone(ModelWrapper._model.params)

            del ModelWrapper._model.params

            # Model string should indicate unset parameters at this point.
            self.assertIsNone(ModelWrapper._model.params)

    def test_edge_cases_visualization_init(self):
        """`visualization.Visualization.init`: Edge Case Validator.

        Tests the behavior of `Visualization.init` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(IndexError):
            # No subplots.
            Visualization("Title", (0, 0))

    def test_edge_cases_visualization_generate_color(self):
        """`visualization.Visualization.generate_color`: Edge Case Validator.

        Tests the behavior of `Visualization.generate_color` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

    def test_edge_cases_visualization_best_fit_lines(self):
        """`visualization.Visualization.best_fit_lines`: Edge Case Validator.

        Tests the behavior of `Visualization.best_fit_lines` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        x = _compose(list,
                     _np.random.uniform)(0.0, 100.0, size=self.data_shape[0])
        """list of float: Random x-values."""
        values = {
            name: _compose(list, _np.random.uniform)(0.0, 100.0,
                                                     size=self.data_shape[0]) \
                  for name in ["observations", "predictions"]
        }
        """:obj:`list of float`: Contains all y-values."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(TypeError):
            # Empty `x`-values.
            v._best_fit_lines([], values, 0)

        with self.assertRaises(ValueError):
            # Empty `y`-values.
            v._best_fit_lines(x, {}, 0)

        with self.assertRaises(IndexError):
            # Suplot index out of range.
            v._best_fit_lines(x, values, 1)

        v.close()

    def test_edge_cases_visualization_empty_subplot(self):
        """`visualization.Visualization.empty_subplot`: Edge Case Validator.

        Tests the behavior of `Visualization.empty_subplot` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(IndexError):
            # Subplot index out of range.
            v._empty_subplot(1)

        v.close()

    def test_edge_cases_visualization_plot_feature(self):
        """`visualization.Visualization.plot_feature`: Edge Case Validator.

        Tests the behavior of `Visualization.plot_feature` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        for ModelWrapper in self.wrappers.values():
            with self.assertRaises(_InvalidFeatureSetError):
                # Empty feature set.
                v._plot_feature(_np.matrix([[]]), Y, 0, ModelWrapper)

            with self.assertRaises(_InvalidObservationSetError):
                # Empty observation set.
                v._plot_feature(X, _np.matrix([[]]), 0, ModelWrapper)

            with self.assertRaises(IndexError):
                # Feature index out of range.
                v._plot_feature(X, Y, self.data_shape[1], ModelWrapper)

        v.close()

    def test_edge_cases_visualization_subplot(self):
        """`visualization.Visualization.subplot`: Edge Case Validator.

        Tests the behavior of `Visualization.subplot` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        x = _compose(list,
                     _np.random.uniform)(0.0, 100.0, size=self.data_shape[0])
        """list of float: Random x-values."""
        values = {
            name: _compose(list, _np.random.uniform)(0.0, 100.0,
                                                     size=self.data_shape[0]) \
                  for name in ["observations", "predictions"]
        }
        """:obj:`list of float`: Contains all y-values."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        for ModelWrapper in self.wrappers.values():
            with self.assertRaises(IndexError):
                # Suplot index out of range.
                v._subplot(1, x, values)

            with self.assertRaises(TypeError):
                # Empty `x`-values.
                v._subplot([], values, 0)

            with self.assertRaises(TypeError):
                # Empty `y`-values.
                v._subplot(x, {}, 0)

        v.close()

    def test_invalid_args_visualization_init(self):
        """`visualization.Visualization.init`: Argument Validator.

        Tests the behavior of `Visualization.init` with invalid argument counts
        and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            Visualization()

        with self.assertRaises(TypeError):
            # Only one argument.
            Visualization("Title")

        with self.assertRaises(TypeError):
            # Invalid kwarg.
            Visualization("Title", (3, 3), key="value")

        with self.assertRaises(TypeError):
            # `None` instead of `subplots`.
            Visualization("Title", None)

        with self.assertRaises(TypeError):
            # Integer for `subplots`.
            Visualization("Title", 3)

        with self.assertRaises(TypeError):
            # Floats instead of `subplots` integers.
            Visualization("Title", (1.2, 3.4))

        with self.assertRaises(ValueError):
            # Negative integers instead of `subplots`.
            Visualization("Title", (-3, -3))

        with self.assertRaises(IndexError):
            # Zero integers instead of row number.
            Visualization("Title", (0, 3))

        with self.assertRaises(IndexError):
            # Zero integers instead of column number.
            Visualization("Title", (3, 0))

    def test_invalid_args_visualization_generate_color(self):
        """`visualization.Visualization.generate_color`: Argument Validator.

        Tests the behavior of `Visualization.generate_color` with invalid
        argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

    def test_invalid_args_visualization_best_fit_lines(self):
        """`visualization.Visualization.best_fit_lines`: Argument Validator.

        Tests the behavior of `Visualization.best_fit_lines` with invalid
        argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        x = _compose(list,
                     _np.random.uniform)(0.0, 100.0, size=self.data_shape[0])
        """list of float: Random x-values."""
        values = {
            name: _compose(list, _np.random.uniform)(0.0, 100.0,
                                                     size=self.data_shape[0]) \
                  for name in ["observations", "predictions"]
        }
        """:obj:`list of float`: Contains all y-values."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(TypeError):
            # No arguments.
            v._best_fit_lines()

        with self.assertRaises(TypeError):
            # Only one argument.
            v._best_fit_lines(x)

        with self.assertRaises(TypeError):
            # Only two arguments.
            v._best_fit_lines(x, values)

        with self.assertRaises(TypeError):
            # Too many arguments.
            v._best_fit_lines(x, values, 0, "extra")

        with self.assertRaises(TypeError):
            # With keyword.
            v._best_fit_lines(x, values, 0, key="value")

        with self.assertRaises(TypeError):
            # `None` instead of list `x`.
            v._best_fit_lines(None, values, 0)

        with self.assertRaises(TypeError):
            # np.matrix instead of list `x`.
            v._best_fit_lines(_np.matrix(x), values, 0)

        with self.assertRaises(TypeError):
            # `None` instead of dict of lists `values`.
            v._best_fit_lines(x, None, 0)

        with self.assertRaises(TypeError):
            value_list = { k: _np.matrix(val) for k, val in values.iteritems() }
            """:obj:`np.matrix`: Lists in `values` mapped to matrices."""

            # Dict of np.matrix instead of dict of lists `values`.
            v._best_fit_lines(x, value_list, 0)

        with self.assertRaises(TypeError):
            # List instead of dict of lists `values`.
            v._best_fit_lines(x, x, 0)

        with self.assertRaises(TypeError):
            # `None` instead of int `subplot`.
            v._best_fit_lines(x, values, None)

        with self.assertRaises(TypeError):
            # Float instead of int `subplot`.
            v._best_fit_lines(x, values, 5.6)

    def test_invalid_args_visualization_empty_subplot(self):
        """`visualization.Visualization.empty_subplot`: Argument Validator.

        Tests the behavior of `Visualization.empty_subplot` with invalid
        argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(TypeError):
            # No arguments.
            v._empty_subplot()

        with self.assertRaises(TypeError):
            # Too many arguments.
            v._empty_subplot(0, 0)

        with self.assertRaises(TypeError):
            # Non-integer index `i`.
            v._empty_subplot(None)

        v.close()

    def test_invalid_args_visualization_plot_feature(self):
        """`visualization.Visualization.plot_feature`: Argument Validator.

        Tests the behavior of `Visualization.plot_feature` with invalid
        argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        X = _random_matrix(self.data_shape)
        """np.matrix: Random-valued feature set."""
        Y = _random_matrix((self.data_shape[0], 1))
        """np.matrix: Random-valued observation set."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(TypeError):
            # No arguments.
            v._plot_feature()

        with self.assertRaises(TypeError):
            # Only one argument.
            v._plot_feature(X)

        with self.assertRaises(TypeError):
            # Only two arguments.
            v._plot_feature(X, Y)

        with self.assertRaises(TypeError):
            # Only three arguments.
            v._plot_feature(X, Y, 0)

        with self.assertRaises(TypeError):
            # `None` instead of model wrapper.
            v._plot_feature(X, Y, 0, None)

        with self.assertRaises(TypeError):
            # `Visualization` instead of `ModelWrapper`.
            v._plot_feature(X, Y, 0, v)

        for ModelWrapper in self.wrappers.values():
            with self.assertRaises(TypeError):
                # Too many arguments.
                v._plot_feature(X, Y, 0, ModelWrapper, "extra")

            with self.assertRaises(TypeError):
                # With keyword.
                v._plot_feature(X, Y, 0, ModelWrapper, key="value")

            with self.assertRaises(_InvalidFeatureSetError):
                # Non-matrix feature set `X`.
                v._plot_feature(None, Y, 0, ModelWrapper)

            with self.assertRaises(_InvalidFeatureSetError):
                # ndarray instead of feature set `X`.
                v._plot_feature(_np.zeros(self.data_shape), Y, 0, ModelWrapper)

            with self.assertRaises(_InvalidObservationSetError):
                # Non-matrix feature set `X`.
                v._plot_feature(X, None, 0, ModelWrapper)

            with self.assertRaises(_InvalidObservationSetError):
                # ndarray instead of feature set `X`.
                v._plot_feature(X, _np.zeros((self.data_shape[0], 1)), 0,
                                ModelWrapper)

            with self.assertRaises(_IncompatibleDataSetsError):
                # Incompatible datasets.
                v._plot_feature(X, _random_matrix((self.data_shape[0] + 1, 1)),
                                0, ModelWrapper)

            with self.assertRaises(TypeError):
                # Non-integer index `feature`.
                v._plot_feature(X, Y, None, ModelWrapper)

            with self.assertRaises(ValueError):
                # Negative integer instead of `feature`.
                v._plot_feature(X, Y, -2, ModelWrapper)

        v.close()

    def test_invalid_args_visualization_subplot(self):
        """`visualization.Visualization.subplot`: Argument Validator.

        Tests the behavior of `Visualization.subplot` with invalid
        argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        x = _compose(list,
                     _np.random.uniform)(0.0, 100.0, size=self.data_shape[0])
        """list of float: Random x-values."""
        values = {
            name: _compose(list, _np.random.uniform)(0.0, 100.0,
                                                     size=self.data_shape[0]) \
                  for name in ["observations", "predictions"]
        }
        """:obj:`list of float`: Contains all y-values."""

        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        with self.assertRaises(TypeError):
            # No arguments.
            v._subplot()

        with self.assertRaises(TypeError):
            # Only one argument.
            v._subplot(x)

        with self.assertRaises(TypeError):
            # Only two arguments.
            v._subplot(x, values)

        with self.assertRaises(TypeError):
            # Too many arguments.
            v._subplot(x, values, 0, "Title", "X-label", "Y-label", True,
                       "extra")

        with self.assertRaises(TypeError):
            # Invalid keyword.
            v._subplot(x, values, 0, key="value")

        with self.assertRaises(TypeError):
            # `None` instead of list `x`.
            v._subplot(None, values, 0)

        with self.assertRaises(TypeError):
            # np.matrix instead of list `x`.
            v._subplot(_np.matrix(x), values, 0)

        with self.assertRaises(TypeError):
            # `None` instead of dict of lists `values`.
            v._subplot(x, None, 0)

        with self.assertRaises(TypeError):
            value_list = { k: _np.matrix(val) for k, val in values.iteritems() }
            """:obj:`np.matrix`: Lists in `values` mapped to matrices."""

            # Dict of np.matrix instead of dict of lists `values`.
            v._subplot(x, value_list, 0)

        with self.assertRaises(TypeError):
            # List instead of dict of lists `values`.
            v._subplot(x, x, 0)

        with self.assertRaises(TypeError):
            # Incompatible x- and y-values.
            values_hat = {
                name: _compose(list, _np.random.uniform)(0.0, 100.0,
                                                         size=self.data_shape[1]) \
                      for name in ["observations", "predictions"]
            }
            """:obj:`list of float`: Contains all y-values."""
            v._subplot(x, values, 0)

        with self.assertRaises(TypeError):
            # `None` instead of int `subplot`.
            v._subplot(x, values, None)

        with self.assertRaises(TypeError):
            # Float instead of int `subplot`.
            v._subplot(x, values, 5.6)

        with self.assertRaises(TypeError):
            # Non-string `title`.
            v._subplot(x, values, 0, title=["hello", "bye"])

        with self.assertRaises(TypeError):
            # Non-string `xlabel`.
            v._subplot(x, values, 0, xlabel=["hello", "bye"])

        with self.assertRaises(TypeError):
            # Non-string `ylabel`.
            v._subplot(x, values, 0, ylabel=["hello", "bye"])

    def test_random_visualization_init(self):
        """`visualization.Visualization.init`: Randomized Validator.

        Tests the behavior of `Visualization.init` by feeding it randomly
        generated arguments.

        Raises:
            AssertionError: If `Visualization.init` needs debugging.

        """
        v = Visualization("Title", (1, 1))
        """Visualization: Plotter instance."""

        # Axes should be a list of length 1.
        self.assertIsInstance(v._ax, list)
        self.assertEqual(len(v._ax), 1)

        v.close()

        combinations = [(2, 3), (3, 2), (6, 1), (1, 6)]
        """list of (int, int): Possible subplot layouts."""

        for i in range(5):
            k = _np.random.randint(0, len(combinations))
            """int: Subplot layout index."""
            v = Visualization("Title", combinations[k])

            # Axes should be a list of length 1.
            self.assertIsInstance(v._ax, list)
            self.assertEqual(len(v._ax), 6)

            v.close()

    def test_random_visualization_generate_color(self):
        """`visualization.Visualization.generate_color`: Randomized Validator.

        Tests the behavior of `Visualization.generate_color` by feeding it
        randomly generated arguments.

        Raises:
            AssertionError: If `Visualization.generate_color` needs debugging.

        """
        for i in range(self.n_tests):
            v = Visualization("Title", (1, 1))
            """Visualization: Plotter instance."""

            color = v._generate_color()
            """(float, float, float): Test input."""

            v.close()

            # Color should be a tuple of floats in the range [0, 1].
            self.assertIsInstance(color, tuple)
            map(_appendargs(self.assertIsInstance, float), color)
            map(_prependargs(self.assertLessEqual, 0.0), color)
            map(_appendargs(self.assertLessEqual, 1.0), color)

    def test_random_visualization_best_fit_lines(self):
        """`visualization.Visualization.best_fit_lines`: Randomized Validator.

        Tests the behavior of `Visualization.best_fit_lines` by feeding it
        randomly generated arguments.

        Raises:
            AssertionError: If `Visualization.best_fit_lines` needs debugging.

        """
        for i in range(self.n_tests):
            v = Visualization("Title", (3, 3))
            """Visualization: Plotter instance."""

            for i in range(9):
                x = _compose(list, _np.random.uniform)(0.0, 100.0,
                                                       size=self.data_shape[0])
                """list of float: Random x-values."""
                values = {
                    name: _compose(list,
                                   _np.random.uniform)(0.0, 100.0,
                                                       size=self.data_shape[0]) \
                          for name in ["observations", "predictions"]
                }
                """:obj:`list of float`: Contains all y-values."""

                lines = v._best_fit_lines(x, values, i)
                """list of matplotlib.lines.Line2D: Test input."""

                # Best-fit lines should be a list of Line2D instances.
                self.assertIsInstance(lines, list)
                map(_appendargs(self.assertIsInstance, _Line2D), lines)

                unique_x = _compose(list, _np.unique)(x)
                """list of float: X-values with duplicates removed."""

                for line in lines:
                    x_data = line.get_xdata()
                    """list of float: X-values in actual best-fit line."""
                    y_data = line.get_ydata()
                    """list of float: Y-values in actual best-fit line."""

                    # Number of x- and y-values in all lines should match number
                    # of unique values in `x`.
                    self.assertEqual(*map(_np.linalg.norm, [x_data, unique_x]))
                    self.assertEqual(*map(len, [y_data, unique_x]))

                    m = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
                    """float: Best-fit line gradient."""
                    b = y_data[0] - m * x_data[0]
                    """float: Y-intercept."""

                    computed_y_vals = map(lambda val: m * val + b, x_data)
                    """list of float: Y-values computed analytically."""

                    # All lines should be rewritable in linear terms.
                    self.assertAlmostEqual(*map(_np.linalg.norm,
                                                [computed_y_vals, y_data]))

            v.close()

    def test_random_visualization_empty_subplot(self):
        """`visualization.Visualization.empty_subplot`: Randomized Validator.

        Tests the behavior of `Visualization.empty_subplot` by feeding it
        randomly generated arguments.

        Raises:
            AssertionError: If `Visualization.empty_subplot` needs debugging.

        """
        for i in range(self.n_tests):
            v = Visualization("Title", (3, 3))
            """Visualization: Plotter instance."""

            for i in range(9):
                v._empty_subplot(i)

                # There should be x- or y-ticks.
                self.assertEqual(len(v._ax[i].get_xticks()), 0)
                self.assertEqual(len(v._ax[i].get_yticks()), 0)

            v.close()

    def test_random_visualization_plot_feature(self):
        """`visualization.Visualization.plot_feature`: Randomized Validator.

        Tests the behavior of `Visualization.plot_feature` by feeding it
        randomly generated arguments.

        Raises:
            AssertionError: If `Visualization.plot_feature` needs debugging.

        """
        for i in range(self.n_tests):
            for ModelWrapper in self.wrappers.values():
                X = _random_matrix(self.data_shape)
                """np.matrix: Random-valued feature set."""
                Y = _random_matrix((self.data_shape[0], 1))
                """np.matrix: Random-valued observation set."""

                v = Visualization("Title", (3, 3))
                """Visualization: Plotter instance."""

                # Intialize model parameters to random values.
                ModelWrapper.model = dict(X=X)

                for i in range(9):
                    x, y, error = v._plot_feature(X, Y, i, ModelWrapper)
                    """(list of float, :obj:`list of float`): X- and y-values to
                    plot."""

                    # `x` should be a list of floats.
                    self.assertIsInstance(x, list)
                    map(_appendargs(self.assertIsInstance, float), x)

                    # Number of `x` values should match number of data points in
                    # `X`.
                    self.assertEqual(len(x), X.shape[0])

                    # `x` values should match all values in `X`.
                    self.assertEqual(*map(_np.linalg.norm, [x, X[:, i]]))

                    # `y` should be a dict.
                    self.assertIsInstance(y, dict)

                    for j, values in _compose(enumerate, y.values)():
                        # `values` should be a list of floats.
                        self.assertIsInstance(values, list)
                        map(_appendargs(self.assertIsInstance, float), values)

                        # Number of values in `values` should match number of
                        # data points in `Y`.
                        self.assertEqual(len(values), Y.shape[0])

                        if j == 0:
                            # Observation values should match all values in `Y`.
                            self.assertEqual(*map(_np.linalg.norm,
                                                  [values, Y[:, 0]]))

                v.close()


Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
