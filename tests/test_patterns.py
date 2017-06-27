import unittest

from hecate.seeds.patterns import ValDict
from hecate.seeds.random import RandInt


class TestValDict(unittest.TestCase):

    def test_constant_value(self):
        d = {'s': 3}
        vd = ValDict(d)
        self.assertEqual(vd['s'], 3, "Wrong constant value.")

    def test_random_value(self):
        d = {'s': RandInt(11, 23)}
        vd = ValDict(d)
        for i in range(10):
            self.assertGreaterEqual(vd['s'], 11, "Wrong random value.")
            self.assertLessEqual(vd['s'], 23, "Wrong random value.")

    def test_multiple_values(self):
        d = {'a': 2, 's': RandInt(11, 23), 'd': 3.3}
        vd = ValDict(d)
        self.assertEqual(vd['a'], 2, "Wrong first constant value.")
        for i in range(10):
            self.assertGreaterEqual(vd['s'], 11, "Wrong random value.")
            self.assertLessEqual(vd['s'], 23, "Wrong random value.")
        self.assertEqual(vd['d'], 3.3, "Wrong second constant value.")

    def test_multiple_dicts(self):
        vd1 = ValDict({'s': 2})
        vd2 = ValDict({'s': RandInt(11, 23)})
        self.assertEqual(vd1['s'], 2, "Wrong first ValDict.")
        for i in range(10):
            self.assertGreaterEqual(vd2['s'], 11, "Wrong second ValDict.")
            self.assertLessEqual(vd2['s'], 23, "Wrong second ValDict.")

    def test_incorrect_key(self):
        vd = ValDict({'s': 2})
        with self.assertRaises(KeyError):
            vd['a']
