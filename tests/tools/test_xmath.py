"""Tests for ``xentica.tools.xmath`` module."""
import unittest

from xentica.tools import xmath


class TestXmath(unittest.TestCase):
    """Tests for xmath functions."""

    def test_popc(self):
        """Test popc function."""
        expr = xmath.popc("x")
        self.assertEqual(expr, "(__popc(x))")

    def test_exp(self):
        """Test exp function."""
        expr = xmath.exp("x")
        self.assertEqual(expr, "(exp(x))")
