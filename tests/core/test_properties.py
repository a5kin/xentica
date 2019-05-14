"""Tests for ``xentica.core.properties`` module."""
import unittest
import binascii

from xentica.core.properties import (
    Property, IntegerProperty, ContainerProperty,
    TotalisticRuleProperty,
)
from xentica.core.exceptions import XenticaException
from examples.game_of_life import GameOfLife
from examples.noisetv import NoiseTV, NoiseTVExperiment


class TestProperty(unittest.TestCase):
    """Tests for ``Property`` class and its children."""

    def test_width(self):
        """Test width calculation."""
        prop = IntegerProperty(max_val=1)
        self.assertEqual(prop.width, 1, "Wrong property width")

    def test_default_bit_width(self):
        """Test default bit width is set to 1."""
        prop = Property()
        self.assertEqual(prop.bit_width, 1, "Wrong default bit width")

    def test_broad_bit_width(self):
        """Test value doesn't fit in any of standard types."""
        prop = IntegerProperty(max_val=10e23)
        self.assertEqual(prop.bit_width, 80, "Wrong bit width")
        self.assertEqual(prop.width, 2, "Wrong width")

    def test_set(self):
        """Test we can set property as class descriptor."""
        # really hypothetic case, just for coverage here
        prop = Property()
        prop.cont = ContainerProperty()
        prop.cont.var_name = "foo"
        prop.cont = 1
        self.assertEqual(type(prop.cont).__name__, "ContainerProperty",
                         "Wrong property's class.")

    def test_unbound(self):
        """Test unbound ``Property`` behavior."""
        prop = Property()
        with self.assertRaises(XenticaException):
            self.assertFalse(prop.declared,
                             "Unbound property declared")
        with self.assertRaises(XenticaException):
            self.assertTrue(prop.coords_declared,
                            "Unbound coords not declared")
        with self.assertRaises(XenticaException):
            cprop = ContainerProperty()
            self.assertFalse(cprop.unpacked,
                             "Unbound property unpacked")

    def test_container_values(self):
        """Test iteration over CA properties."""
        model = GameOfLife
        props = [v for v in model.main.values()]
        self.assertEqual(len(props), 1, "Wrong number of properties")

    def test_random_property(self):
        """Test ``RandomProperty`` behavior."""
        model = NoiseTV(NoiseTVExperiment)
        for _ in range(100):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(2773894957, checksum, "Wrong field checksum.")

    def test_pure_totalistic(self):
        """Test pure totalistic rule behavior."""
        prop = TotalisticRuleProperty(outer=False)
        with self.assertRaises(XenticaException):
            prop.is_sustained(8)
        with self.assertRaises(XenticaException):
            prop.is_born(8)
        prop.var_name = "rule"
        correct_exp = "((rule >> 4) & 1)"
        self.assertEqual(str(prop.next_val(1, 3)), correct_exp)

    def test_outer_totalistic(self):
        """Test outer totalistic rule behavior."""
        prop = TotalisticRuleProperty(outer=True)
        prop.var_name = "rule"
        correct_exp = "(((rule & 261120) >> 17) & 1)"
        self.assertEqual(str(prop.is_sustained(8)), correct_exp)
        correct_exp = "(((rule & 510) >> 8) & 1)"
        self.assertEqual(str(prop.is_born(8)), correct_exp)
        correct_exp = "((rule >> 12) & 1)"
        self.assertEqual(str(prop.next_val(1, 3)), correct_exp)
