import unittest as _unittest

from testaugmentors import Test as _AugmentorsTest
from testgeneral import Test as _GeneralUtilsTest
from testlinalg import Test as _LinearAlgebraUtilsTest
from testloss import Test as _LossTest
from teststats import Test as _StatsTest


__all__ = ['UtilsTestSuite']

_tests = [_AugmentorsTest, _GeneralUtilsTest, _LinearAlgebraUtilsTest,
          _LossTest, _StatsTest]

UtilsTestSuite = _unittest.TestSuite(_tests)
