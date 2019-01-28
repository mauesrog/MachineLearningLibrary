"""General Utilities Testing Module.

"""
import unittest as _unittest
import numpy as _np
from math import sqrt as _sqrt

from common.test_cases.module_test_case import ModuleTestCase as _ModuleTestCase
from general import compose


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

    def test_edge_cases_compose(self):
        """`general.compose`: Edge Case Validator.

        Tests the behavior of `compose` with edge cases.

        Raises:
            Exception: If at least one `Exception` raised is not of the expected
                kind.

        """
        pass

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

    def test_random_compose(self):
        """`general.compose`: Randomized Validator.

        Tests the behavior of `compose` by feeding it randomly generated arguments.

        Raises:
            AssertionError: If `compose` needs debugging.

        """
        negate = lambda x: -1.0 * x
        """callable: Negates nuemeric input."""
        functions = str, negate, _sqrt, abs, sum
        """tuple of callable: Functions that will get composed during each
        random iteration of the test."""

        # Compose with no arguments should return `None`.
        self.assertIsNone(compose())

        for i in range(0, self.n_tests):
            args = (_np.random.uniform(0.0, 100.0, size=self.n),)

            composed_fn = compose(*functions)
            """callable: Test input."""

            # Composition should be a function.
            self.assertEqual(type(composed_fn), type(compose))

            result = composed_fn(*args)
            """str: Composition output."""

            # Programatic composition result should match manual composition
            # output.
            self.assertEqual(result, str(negate(_sqrt(abs(sum(*args))))))

Test = _unittest.TestLoader().loadTestsFromTestCase(_Test)
"""TestSuite: General utilities testing suite."""
