"""Tests for ``xentica.core.variables`` module."""
import unittest

from xentica.core.variables import Variable, IntegerVariable
from xentica.core.exceptions import XenticaException
from xentica import core
from xentica import seeds


class InvertCA(core.CellularAutomaton):
    """CA that inverts each cell's value each step."""
    state = core.IntegerProperty(max_val=1)
    # vars should be declared at the class level,
    # in order to use them in direct assigns
    intvar = IntegerVariable()

    class Topology:
        """Most common topology."""
        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    def emit(self):
        """Do nothing at emit phase."""
        self.intvar = self.main.state
        self.buffers[0].state = self.intvar

    def absorb(self):
        """Invert cell's value."""
        self.intvar = self.buffers[0].state
        self.main.state = -self.intvar
        var_list = [
            IntegerVariable(name="self_var"),
            IntegerVariable(),
        ]
        var_list[1] += 1

    def color(self):
        """Do nothing, no color processing required."""


class InvertExperiment(core.Experiment):
    """Test experiment for ``InvertCA``."""

    word = "INSIDE OUT"
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
        myvar1 = IntegerVariable()
        self.assertEqual(str(myvar1), "myvar1", "Wrong variable name.")
        self.assertEqual(myvar1.code, "myvar1", "Wrong variable name.")
        myvar2 = IntegerVariable(1)
        self.assertEqual(str(myvar2), "myvar2", "Wrong variable name.")
        var_list = [
            IntegerVariable(1, name="myvar3"),
            IntegerVariable(name="myvar4"),
            IntegerVariable(2),
            IntegerVariable(),
        ]
        self.assertEqual(var_list[0].var_name, "myvar3",
                         "Wrong variable name.")
        self.assertEqual(var_list[1].var_name, "myvar4",
                         "Wrong variable name.")
        self.assertEqual(var_list[2].var_name, "var",
                         "Wrong variable name.")
        self.assertEqual(var_list[3].var_name, "var",
                         "Wrong variable name.")

    def test_descriptor(self):
        """Test ``Variable`` acting as class descriptor."""

        model = InvertCA(InvertExperiment)
        correct_emit = """
unsigned char _cell_state;
_cell_state = (_cell) & 1;
unsigned int intvar = 0;
intvar = _cell_state;
unsigned char _bcell_state0;
_bcell_state0 = intvar;
unsigned char _bcell0;
_bcell0 = ((unsigned char) _bcell_state0 & 1);
fld[i + n * 8] = _bcell0;
"""
        correct_absorb = """
unsigned char _bcell_state0;
_bcell_state0 = (_bcell0) & 1;
unsigned int intvar = 0;
intvar = _bcell_state0;
unsigned char _cell_state;
_cell_state = (-(intvar));
unsigned char _cell = fld[i];
unsigned int var = 0;
var += 1;
_cell = ((unsigned char) _cell_state & 1);
fld[i] = _cell;
"""
        self.assertIn(correct_emit, model.cuda_source,
                      "Wrong code for emit().")
        self.assertIn(correct_absorb, model.cuda_source,
                      "Wrong code for absorb().")

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
