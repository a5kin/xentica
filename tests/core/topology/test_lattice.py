import unittest

from xentica.core.exceptions import XenticaException
from xentica.core.topology.lattice import (
    OrthogonalLattice,
)


class TestOrthogonalLattice(unittest.TestCase):

    def test_incorrect_dimensions(self):
        lattice = OrthogonalLattice()
        with self.assertRaises(XenticaException):
            lattice.dimensions = 0
