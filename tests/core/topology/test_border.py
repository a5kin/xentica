import unittest

from hecate.core.base import HecateException
from hecate.core.topology.border import (
    TorusBorder,
)


class TestTorusBorder(unittest.TestCase):

    def test_incorrect_dimensions(self):
        border = TorusBorder()
        with self.assertRaises(HecateException):
            border.set_dimensions(0)
