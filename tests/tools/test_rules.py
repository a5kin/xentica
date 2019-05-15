"""Tests for ``xentica.tools.rules`` module."""
import unittest

from xentica.tools.rules import LifeLike


class TestLifelike(unittest.TestCase):
    """Tests for Lifelike rules helpers."""

    def test_golly(self):
        """Test conversion from Golly and vice versa."""
        rule_str = "B3/S23"
        rule_int = LifeLike.golly2int(rule_str)
        self.assertEqual(rule_int, 6152)
        rule_converted = LifeLike.int2golly(rule_int)
        self.assertEqual(rule_converted, rule_str)
