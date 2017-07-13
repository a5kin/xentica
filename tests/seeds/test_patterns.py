import unittest
import binascii

import numpy as np

from hecate.seeds.patterns import ValDict
from hecate.seeds.patterns import BigBang, PrimordialSoup
from hecate.seeds.random import RandInt, LocalRandom


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

    def test_items(self):
        d = {'a': 2, 's': RandInt(11, 23), 'd': 3.3}
        vd = ValDict(d)
        for k, v in vd.items():
            if k == 'a':
                self.assertEqual(v, 2, "Wrong constant value.")
            elif k == 'b':
                self.assertGreaterEqual(v, 11, "Wrong random value.")
                self.assertLessEqual(v, 23, "Wrong random value.")

    def test_incorrect_access(self):
        with self.assertRaises(NotImplementedError):
            vd = ValDict({})
            vd['a'] = 2


class TestPatternBase(unittest.TestCase):

    def index_to_coord(self, i):
        return (i % 100, i // 100)

    def pack_state(self, state):
        return state['s']


class TestBigBang(TestPatternBase):

    def test_2d(self):
        pos = (32, 20)
        size = (10, 10)
        vals = {'s': RandInt(0, 1)}
        bb = BigBang(pos=pos, size=size, vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, 10000, (100, 100),
                    self.index_to_coord, self.pack_state)
        for i in np.where(cells == 1)[0]:
            x, y = self.index_to_coord(i)
            self.assertGreaterEqual(x, pos[0], "Wrong right bound.")
            self.assertGreaterEqual(y, pos[1], "Wrong upper bound.")
            self.assertLessEqual(x, size[0] + pos[0], "Wrong left bound.")
            self.assertLessEqual(y, size[1] + pos[1], "Wrong lower bound.")

    def test_random_size(self):
        vals = {'s': RandInt(0, 1)}
        bb = BigBang(pos=(10, 10), vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, 10000, (100, 100),
                    self.index_to_coord, self.pack_state)

    def test_random_pos(self):
        vals = {'s': RandInt(0, 1)}
        bb = BigBang(size=(10, 10), vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, 10000, (100, 100),
                    self.index_to_coord, self.pack_state)

    def test_random_full(self):
        vals = {'s': RandInt(0, 1)}
        bb = BigBang(vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, 10000, (100, 100),
                    self.index_to_coord, self.pack_state)


class TestPrimordialSoup(TestPatternBase):

    def test_2d(self):
        vals = {'s': RandInt(0, 1)}
        seed = PrimordialSoup(vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        seed.random = LocalRandom("test")
        seed.generate(cells, 10000, (100, 100),
                      self.index_to_coord, self.pack_state)
        self.assertEqual(binascii.crc32(cells[:10000]), 2251764292,
                         "Wrong field checksum.")
