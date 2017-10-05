import unittest

from hecate.core.base import HecateException
from hecate.core.topology.neighborhood import (
    MooreNeighborhood,
)


class TestMooreNeighborhood(unittest.TestCase):

    def test_incorrect_dimensions(self):
        neighborhood = MooreNeighborhood()
        with self.assertRaises(HecateException):
            neighborhood.dimensions = 0
