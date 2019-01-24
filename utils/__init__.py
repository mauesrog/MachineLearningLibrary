import unittest as _unittest

from augmentors import Test as _AugmentorsTest
from general import Test as _GeneralUtilsTest
from linalg import Test as _LinearAlgebraUtilsTest
from loss import Test as _LossTest
from stats import Test as _StatsTest


__all__ = ['UtilsTestSuite']

_tests = [_AugmentorsTest, _GeneralUtilsTest, _LinearAlgebraUtilsTest,
          _LossTest, _StatsTest]

UtilsTestSuite = _unittest.TestSuite(_tests)
