import unittest

from config import verbosity
from utils import UtilsTestSuite
from models import ModelsTestSuite
from __init__ import LearnerTestSuite


if __name__ == "__main__":
    print "--Utilities ->"
    unittest.TextTestRunner(verbosity=verbosity).run(UtilsTestSuite)

    print "\n\n--Models ->"
    unittest.TextTestRunner(verbosity=verbosity).run(ModelsTestSuite)

    print "\n\n--Learner ->"
    unittest.TextTestRunner(verbosity=verbosity).run(LearnerTestSuite)
