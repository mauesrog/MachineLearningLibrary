import unittest as _unittest

from testlearner import Test as _LearnerTest


__all__ = ['LearnerTestSuite']
_tests = [_LearnerTest]

LearnerTestSuite = _unittest.TestSuite(_tests)
