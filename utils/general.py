"""General Model Utilities.

Defines general methods needed by models.

"""
import unittest as _unittest
import numpy as _np
from math import sqrt as _sqrt

from test import ModuleTestCase as _ModuleTestCase

def compose(*fns):
    """Function Composer.

    Creates a callable composition with the functions provided. Useful for
    readability and decluttering.

    Example:
        `compose(f, e, d, c, b, a)(*args)` is equivalent to
        `f(e(d(c(b(a(*args))))))`.

    Args:
        *fns (tuple of callable): Functions to compose.

    Returns:
        callable: The function composition capable of receiving *args, `None`
            if no functions were provided.

    """
    # Needs at least one function to compose
    if len(fns) == 0:
        return None

    return lambda *args: _compose_helper(list(fns), args)

def _compose_helper(fns, args):
    """Function Composer Helper.

    Calls the given functions in order with the arguments provided (which will
    get consumed by the very first function).

    Args:
        fns (tuple of callable): Functions to compose, provided in the exact
            order they should get consumed.
        args: Arguments to feed the very first function i.e. `fns[0]`.

    Returns:
        The result of composing all functions with the given arguments, `None`
        if no functions were provided.

    Raises:
        TypeError: If `fns` contains a non-callable element.

    """
    result = fns.pop()(*args)
    """Keeps track of the running result of the function composition, started
    by feeding the given arguments to the first functions."""

    # Iterate until all functions have been consumed.
    while len(fns):
        result = fns.pop()(result)

    return result


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
