import unittest

from hecate.core.base import HecateException
from hecate.core.topology.lattice import (
    OrthogonalLattice,
)


class TestOrthogonalLattice(unittest.TestCase):

    def test_incorrect_dimensions(self):
        lattice = OrthogonalLattice()
        with self.assertRaises(HecateException):
            lattice.set_dimensions(0)
