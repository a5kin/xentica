import unittest

from hecate.core.variables import IntegerVariable


class TestVariable(unittest.TestCase):

    def test_integer(self):
        IntegerVariable()
