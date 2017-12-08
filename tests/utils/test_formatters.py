"""Tests for ``xentica.utils.formatters`` module."""
import unittest

from xentica.utils.formatters import sizeof_fmt


class TestSizeFormatter(unittest.TestCase):
    """Tests for size formatter helper."""

    def test_less_kilo(self):
        """Test less than Kilo rendering as plain values."""
        val = sizeof_fmt(345)
        self.assertEqual(val, "345", "Less Kilo format is incorrect.")

    def test_kilo(self):
        """Test numbers between 10e3 and 10e6 has K postix."""
        val = sizeof_fmt(34567)
        self.assertEqual(val, "34.57K", "Kilo format is incorrect.")

    def test_peta(self):
        """Test numbers between 10e15 and 10e18 has P postix."""
        val = sizeof_fmt(34567 * (10 ** 12))
        self.assertEqual(val, "34.57P", "Peta format is incorrect.")

    def test_yokta(self):
        """Test numbers larger than 10e24 has Y postix."""
        val = sizeof_fmt(34567 * (10 ** 21))
        self.assertEqual(val, "34.57Y", "Yokta format is incorrect.")
