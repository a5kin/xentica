"""Tests for ``xentica.core.variables`` module."""
import unittest

from xentica.core.variables import Variable, IntegerVariable
from xentica.core.exceptions import XenticaException
from xentica import core
from xentica import seeds


class InvertCA(core.CellularAutomaton):
    """CA that inverts each cell's value each step."""
    state = core.IntegerProperty(max_val=1)

    class Topology:
        """Most common topology."""
        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    def emit(self):
        """Do nothing at emit phase."""

    def absorb(self):
        """Invert cell's value."""
        # really weird way to declare variables
        # but at least it work with current system
        self.__class__.intvar = IntegerVariable()
        self.intvar = self.main.state
        self.main.state = -self.intvar

    def color(self):
        """Do nothing, no color processing required."""


class InvertExperiment(core.Experiment):
    """Test experiment for ``InvertCA``."""

    size = (64, 36, )
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


class TestVariable(unittest.TestCase):
    """Tests for ``Variable`` class and its children."""

    def test_integer(self):
        """Test ``IntegerVariable`` initialization."""
        var = IntegerVariable()
        self.assertEqual(str(var), "var", "Wrong variable name.")

    def test_descriptor(self):
        """Test ``Variable`` acting as class descriptor."""

        self.assertEqual(InvertCA.__name__, "InvertCA",
                         "Wrong class name.")

    def test_illegal_assign(self):
        """Test illegal assign to ``DeferredExpression``."""

        with self.assertRaises(XenticaException):
            class BrokenCA(InvertCA):
                """Class for broken expressions testing."""

                def emit(self):
                    """Try to assign to DeferredExpression"""
                    deferred_exp = 1 + self.main.state
                    deferred_exp += 1

            BrokenCA(InvertExperiment)

    def test_no_init_val(self):
        """Test initialization without initial value."""
        with self.assertRaises(XenticaException):
            Variable()
