import unittest as _unittest

from testlinear import Test as _LinearModelTest


__all__ = ['ModelsTestSuite']

_tests = [_LinearModelTest]

ModelsTestSuite = _unittest.TestSuite(_tests)
