"""General Utilities Testing Module.

Attributes:
    Test (TestSuite): General utilities testing suite.

"""
import unittest as _unittest
import numpy as _np
from math import sqrt as _sqrt

from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from general import appendargs, compose, prependargs


class _Test(_ModuleTestCase):
    """General Model Utilities Unit Tester.

    Runs tests for all general model utilties:
        - `compose`

    Attributes:
        label (str): Identifier for super class to generate custome test
            docstrings according to the general model utilities module. See base
            class.
        n (int): Number of data points to use in any test e.g. in random tests.
        n_tests (int): Number of tests to run for any test e.g. in random tests.
        name (str): Name of current module. See base class.

    """
    def setUp(self):
        """General Utilities Testing Configuration.

        Sets up the necessary information to begin testing.

        """
        self.label = '`general`'
        self.n = 100
        self.n_tests = 20
        self.name = __name__

    def test_edge_cases_appendargs(self):
        """`general.appendargs`: Edge Case Validator.

        Tests the behavior of `appendargs` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(AttributeError):
            # No arguments to append.
            appendargs(sum)([2])

    def test_edge_cases_compose(self):
        """`general.compose`: Edge Case Validator.

        Tests the behavior of `compose` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

    def test_invalid_args_appendargs(self):
        """`general.appendargs`: Argument Validator.

        Tests the behavior of `appendargs` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            appendargs()([2])

        with self.assertRaises(TypeError):
            # Non-function argument.
            appendargs(None, 2)([2])

        with self.assertRaises(TypeError):
            # With **kwargs.
            appendargs(sum, 2, key="value")([2])

    def test_invalid_args_compose(self):
        """`general.compose`: Argument Validator.

        Tests the behavior of `compose` with invalid argument counts and values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            compose()("arg")

        with self.assertRaises(TypeError):
            # Non-function arguments.
            compose(map, reduce, filter, 123)("arg")

        with self.assertRaises(TypeError):
            # With **kwargs.
            compose(map, reduce, key1="key1", key2="key2")("arg")

    def test_invalid_args_prependargs(self):
        """`general.prependargs`: Argument Validator.

        Tests the behavior of `prependargs` with invalid argument counts and
        values.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        with self.assertRaises(TypeError):
            # No arguments.
            prependargs()([2])

        with self.assertRaises(TypeError):
            # Non-function argument.
            prependargs(None, 2)([2])

        with self.assertRaises(TypeError):
            # With **kwargs.
            prependargs(sum, 2, key="value")([2])

    def test_random_appendargs(self):
        """`general.appendargs`: Randomized Validator.

        Tests the behavior of `appendargs` by feeding it randomly generated arguments.

        Raises:
            AssertionError: If `appendargs` needs debugging.

        """
        concat = lambda s1, s2: s1 + s2
        """callable: Appends one string to the other."""

        # appendargs with no arguments should return `None`.
        self.assertIsNone(appendargs(None))

        for i in range(self.n_tests):
            c_arg = compose(str, _np.random.uniform)(0.0, 100.0, size=self.n)
            """list of float: Argument to be appended."""
            arg = compose(str, _np.random.uniform)(0.0, 100.0, size=self.n)
            """list of float: Test argument."""

            target = arg + c_arg
            """float: Expected output from appendargsd function."""

            adder = appendargs(concat, c_arg)
            """callable: Test input."""

            # Adder should be a function.
            self.assertIsInstance(adder, type(compose))

            result = adder(arg)
            """str: Adder output."""

            # Programatic adder result should match manual sum.
            self.assertEqual(result, target)

    def test_random_compose(self):
        """`general.compose`: Randomized Validator.

        Tests the behavior of `compose` by feeding it randomly generated arguments.

        Raises:
            AssertionError: If `compose` needs debugging.

        """
        def special_sum(*args, **kwargs):
            return sum(list(args) + kwargs.values())

        negate = lambda x: -1.0 * x
        """callable: Negates nuemeric input."""

        functions = str, negate, _sqrt, abs, special_sum
        """tuple of callable: Functions that will get composed during each
        random iteration of the test."""

        # Compose with no arguments should return `None`.
        self.assertIsNone(compose())

        for i in range(self.n_tests):
            args = _np.random.uniform(0.0, 100.0, size=self.n)
            kwargs = { str(k): k for k in _np.random.uniform(0.0, 100.0, size=self.n) }

            target = str(negate(_sqrt(abs(sum(args + kwargs.values())))))
            """str: Expected output from composed function."""

            composed_fn = compose(*functions)
            """callable: Test input."""

            # Composition should be a function.
            self.assertEqual(type(composed_fn), type(compose))

            result = composed_fn(*args, **kwargs)
            """str: Composition output."""

            # Programatic composition result should match manual composition
            # output.
            self.assertEqual(result, target)

    def test_random_prependargs(self):
        """`general.prependargs`: Randomized Validator.

        Tests the behavior of `prependargs` by feeding it randomly generated
        arguments.

        Raises:
            AssertionError: If `prependargs` needs debugging.

        """
        concat = lambda s1, s2: s1 + s2
        """callable: Appends one string to the other."""

        # prependargs with no arguments should return `None`.
        self.assertIsNone(prependargs(None))

        for i in range(self.n_tests):
            c_arg = compose(str, _np.random.uniform)(0.0, 100.0, size=self.n)
            """list of float: Argument to be appended."""
            arg = compose(str, _np.random.uniform)(0.0, 100.0, size=self.n)
            """list of float: Test argument."""

            target = c_arg + arg
            """float: Expected output from prependargsd function."""

            adder = prependargs(concat, c_arg)
            """callable: Test input."""

            # Adder should be a function.
            self.assertIsInstance(adder, type(compose))

            result = adder(arg)
            """str: Adder output."""

            # Programatic adder result should match manual sum.
            self.assertEqual(result, target)

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
