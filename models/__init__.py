import unittest as _unittest

from linear import LinearModel as _LinearModel

from testlinear import Test as _LinearModelTest
from testmodelwrapper import Test as _ModelWrapperTest


__all__ = ['ModelsTestSuite']

_tests = [_LinearModelTest, _ModelWrapperTest]

ModelsTestSuite = _unittest.TestSuite(_tests)
