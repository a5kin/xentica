import unittest

from xentica.utils.formatters import sizeof_fmt


class TestSizeFormatter(unittest.TestCase):

    def test_less_kilo(self):
        val = sizeof_fmt(345)
        self.assertEqual(val, "345", "Less Kilo format is incorrect.")

    def test_kilo(self):
        val = sizeof_fmt(34567)
        self.assertEqual(val, "34.57K", "Kilo format is incorrect.")

    def test_peta(self):
        val = sizeof_fmt(34567 * (10 ** 12))
        self.assertEqual(val, "34.57P", "Peta format is incorrect.")

    def test_yokta(self):
        val = sizeof_fmt(34567 * (10 ** 21))
        self.assertEqual(val, "34.57Y", "Yokta format is incorrect.")
