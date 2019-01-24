import unittest
from utils import UtilsTestSuite
from models import ModelsTestSuite
from config import verbosity

if __name__ == "__main__":
    print "--Utilities ->"
    unittest.TextTestRunner(verbosity=verbosity).run(UtilsTestSuite)

    print "\n\n--Models ->"
    unittest.TextTestRunner(verbosity=verbosity).run(ModelsTestSuite)
