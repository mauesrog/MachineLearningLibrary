"""Module Unit Test Template.

Provides a general way for all modules to perform unit testing.

Attributes:
    See `config.testing`.

"""
import sys as _sys
from unittest import TestCase as _TestCase
import re as _re

from config import testing_defaults as _testing_defaults

_DEFAULT_ZERO_CUTOFF = _testing_defaults["zero_cutoff"]


class ModuleTestCase(_TestCase):
    """Module Test Case.

    Abstracts some logic from the per-module TestCase instances e.g. checks for
    documentation.

    Attributes:
        zero_cutoff (float): Largest number to treat as zero.

    """
    def __init__(self, *args, **kwargs):
        """Module Test Case Constructor.

        """
        self.zero_cutoff = _DEFAULT_ZERO_CUTOFF
        super(ModuleTestCase, self).__init__(*args, **kwargs)

    def assertAlmostEqual(self, v1, v2):
        self.assertLessEqual(abs(v1 - v2), self.zero_cutoff)

    def test_documentation(self):
        """Documentation Validator.

        Ensures that there exist non-empty docstrings for `self`, the module
        itself, and all module-level public declarations (e.g. methods,
        variables).

        Raises:
            AssertionError: If there is at least one docstring missing from the
                module's documentation.

        """
        module = _sys.modules[self.name or __name__]
        """module: This module if `self` is a direct instance of ModuleTestCase,
        the inheriting child."""

        try:
            self.assertTrue(type(module.__doc__) == str and \
                            len(module.__doc__) > 0)
        except AssertionError as e:
            raise AssertionError("No module-level documentation found!")

        testcase_regex = _re.compile(r"^_[a-zA-Z]*Test$")
        """SRE_Pattern: Identifies testing cases."""
        test_regex = _re.compile(r"^[a-zA-Z]*Test$")
        """SRE_Pattern: Identifies testing suites."""

        privacy_checker = lambda s: s[0] != '_' and not test_regex.match(s)
        """callable: Checks whether the given attribute name is private. Note:
        Also excludes `Test` from consideration since its not a testable
        attribute."""

        attrs = dir(module)
        """list of str: Module attribute names."""

        fn_names = filter(privacy_checker, attrs) + \
                   filter(testcase_regex.match, attrs)
        """list of str: Public declarations of `module` a.k.a. the names to test
        for docstrings."""

        for fn in fn_names:
            export = getattr(module, fn)  #: Variable with docstring.

            try:
                self.assertTrue(type(export.__doc__) == str and \
                                len(export.__doc__) > 0)
            except AssertionError as e:
                raise AssertionError("No documentation found for '%s'!" % fn)


def _compose(*fns):
    """Function Composer.

    See `utils.general.compose`.

    Note:
        Added to avoid circular imports.

    """
    # Needs at least one function to compose
    if len(fns) == 0:
        return None

    return lambda *args: _compose_helper(list(fns), args)

def _compose_helper(fns, args):
    """Function Composer Helper.

    See `utils.general.compose`.

    Note:
        Added to avoid circular imports.

    """
    result = fns.pop()(*args)
    """Keeps track of the running result of the function composition, started
    by feeding the given arguments to the first functions."""

    # Iterate until all functions have been consumed.
    while len(fns):
        result = fns.pop()(result)

    return result
