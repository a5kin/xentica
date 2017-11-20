import unittest

from xentica.core.base import XenticaException
from xentica.core.topology.border import (
    TorusBorder,
)


class TestTorusBorder(unittest.TestCase):

    def test_incorrect_dimensions(self):
        border = TorusBorder()
        with self.assertRaises(XenticaException):
            border.dimensions = 0
