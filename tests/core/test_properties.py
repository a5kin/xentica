"""Tests for ``xentica.core.properties`` module."""
import unittest

from xentica.core.properties import (
    Property, IntegerProperty, ContainerProperty
)
from examples.game_of_life import GameOfLife


class TestProperty(unittest.TestCase):
    """Tests for ``Property`` class and its children."""

    def test_width(self):
        """Test width calculation."""
        p = IntegerProperty(max_val=1)
        self.assertEqual(p.width, 1, "Wrong property width")

    def test_default_bit_width(self):
        """Test default bit width is set to 1."""
        p = Property()
        self.assertEqual(p.bit_width, 1, "Wrong default bit width")

    def test_set(self):
        """Test we can set property as class descriptor."""
        # really hypothetic case, just for coverage here
        self.p = Property()
        self.p.r = ContainerProperty()
        self.p.r.var_name = "r"
        self.p.r = 1

    def test_unbound(self):
        """Test unbound ``Property`` default flags values."""
        self.p = Property()
        self.assertFalse(self.p._declared,
                         "Unbound property declared")
        self.assertTrue(self.p._coords_declared,
                        "Unbound coords not declared")
        self.cp = ContainerProperty()
        self.assertFalse(self.cp._unpacked,
                         "Unbound property unpacked")

    def test_container_values(self):
        """Test iteration over CA properties."""
        ca = GameOfLife
        props = [v for v in ca.main.values()]
        self.assertEqual(len(props), 1, "Wrong number of properties")
