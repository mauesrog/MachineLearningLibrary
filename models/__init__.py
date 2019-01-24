import unittest as _unittest

from linear import Test as _LinearModelTest


__all__ = ['ModelsTestSuite']

ModelsTestSuite = _unittest.TestSuite([_LinearModelTest])
