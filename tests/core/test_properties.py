"""Tests for ``xentica.core.properties`` module."""
import unittest
import binascii

from xentica.core.properties import (
    Property, IntegerProperty, ContainerProperty
)
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
        self.assertEqual(prop.width, 3, "Wrong width")

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
        """Test unbound ``Property`` default flags values."""
        prop = Property()
        self.assertFalse(prop.declared,
                         "Unbound property declared")
        self.assertTrue(prop.coords_declared,
                        "Unbound coords not declared")
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

