import unittest

from xentica.core.base import XenticaException
from xentica.core.topology.neighborhood import (
    MooreNeighborhood,
)


class TestMooreNeighborhood(unittest.TestCase):

    def test_incorrect_dimensions(self):
        neighborhood = MooreNeighborhood()
        with self.assertRaises(XenticaException):
            neighborhood.dimensions = 0
