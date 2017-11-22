import unittest

from xentica.core.variables import IntegerVariable
from xentica import core


class TestVariable(unittest.TestCase):

    def test_integer(self):
        IntegerVariable()

    def test_descriptor(self):
        class InvertCA(core.CellularAutomaton):
            state = core.IntegerProperty(max_val=1)

            class Topology:
                dimensions = 2
                lattice = core.OrthogonalLattice()
                neighborhood = core.MooreNeighborhood()
                border = core.TorusBorder()

            def emit(self):
                pass

            def absorb(self):
                # really weird way to declare variables
                # but at least it work with current system
                self.__class__.intvar = IntegerVariable()
                self.intvar = self.main.state
                self.main.state = -self.intvar

            def color(self):
                pass
