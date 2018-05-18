"""Tests for ``xentica.seeds.patterns`` module."""
import unittest
import binascii

import numpy as np

from xentica.seeds.patterns import ValDict
from xentica.seeds.patterns import BigBang, PrimordialSoup
from xentica.seeds.random import RandInt, LocalRandom
from examples.game_of_life import GameOfLife, GOLExperiment


class TestValDict(unittest.TestCase):
    """Tests for ``ValDict`` class."""

    def test_constant_value(self):
        """Test constant value storing."""
        d = {'s': 3}
        vd = ValDict(d)
        self.assertEqual(vd['s'], 3, "Wrong constant value.")

    def test_random_value(self):
        """Test random value storing."""
        d = {'s': RandInt(11, 23)}
        vd = ValDict(d)
        for i in range(10):
            self.assertGreaterEqual(vd['s'], 11, "Wrong random value.")
            self.assertLessEqual(vd['s'], 23, "Wrong random value.")

    def test_multiple_values(self):
        """Test multiple values storing."""
        d = {'a': 2, 's': RandInt(11, 23), 'd': 3.3}
        vd = ValDict(d)
        self.assertEqual(vd['a'], 2, "Wrong first constant value.")
        for i in range(10):
            self.assertGreaterEqual(vd['s'], 11, "Wrong random value.")
            self.assertLessEqual(vd['s'], 23, "Wrong random value.")
        self.assertEqual(vd['d'], 3.3, "Wrong second constant value.")

    def test_multiple_dicts(self):
        """Test different dicts keeping different values."""
        vd1 = ValDict({'s': 2})
        vd2 = ValDict({'s': RandInt(11, 23)})
        self.assertEqual(vd1['s'], 2, "Wrong first ValDict.")
        for i in range(10):
            self.assertGreaterEqual(vd2['s'], 11, "Wrong second ValDict.")
            self.assertLessEqual(vd2['s'], 23, "Wrong second ValDict.")

    def test_incorrect_key(self):
        """Test incorrect key access."""
        vd = ValDict({'s': 2})
        with self.assertRaises(KeyError):
            vd['a']

    def test_items(self):
        """Test iteration over items."""
        d = {'a': 2, 's': RandInt(11, 23), 'd': 3.3}
        vd = ValDict(d)
        for k, v in vd.items():
            if k == 'a':
                self.assertEqual(v, 2, "Wrong constant value.")
            elif k == 's':
                self.assertGreaterEqual(v, 11, "Wrong random value.")
                self.assertLessEqual(v, 23, "Wrong random value.")

    def test_incorrect_access(self):
        """Test item setting."""
        with self.assertRaises(NotImplementedError):
            vd = ValDict({})
            vd['a'] = 2


class TestPatternBase(unittest.TestCase):
    """Base class for patterns testing."""

    def index_to_coord(self, i):
        """Emulate ``CellularAutomaton.index_to_coord`` behavior."""
        return (i % 100, i // 100)

    def pack_state(self, state):
        """Emulate ``CellularAutomaton.pack_state`` behavior."""
        return state['s']


class TestExperiment(GOLExperiment):
    """Regular experiment for tests."""
    size = (100, 100, )
    seed = BigBang(
        pos=(32, 20),
        size=(10, 10),
        vals={
            "state": RandInt(0, 1),
        }
    )


class TestBigBang(TestPatternBase):
    """Tests for ``BigBang`` class."""

    def test_2d(self):
        """Test all cells are inside region after generation."""
        bsca = GameOfLife(TestExperiment)
        cells = bsca.cells_gpu.get()[:bsca.cells_num]
        pos = (32, 20)
        size = (10, 10)
        for i in np.where(cells == 1)[0]:
            x, y = self.index_to_coord(i)
            self.assertGreaterEqual(x, pos[0], "Wrong right bound.")
            self.assertGreaterEqual(y, pos[1], "Wrong upper bound.")
            self.assertLessEqual(x, size[0] + pos[0], "Wrong left bound.")
            self.assertLessEqual(y, size[1] + pos[1], "Wrong lower bound.")

    def test_random_size(self):
        """Test omitted size behavior."""
        bsca = GameOfLife(TestExperiment)
        vals = {'state': RandInt(0, 1)}
        bb = BigBang(pos=(10, 10), vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, bsca)

    def test_random_pos(self):
        """Test omitted position behavior."""
        bsca = GameOfLife(TestExperiment)
        vals = {'state': RandInt(0, 1)}
        bb = BigBang(size=(10, 10), vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, bsca)

    def test_random_full(self):
        """Test omitted size and position behavior."""
        bsca = GameOfLife(TestExperiment)
        vals = {'state': RandInt(0, 1)}
        bb = BigBang(vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, bsca)

    def test_wrong_pos(self):
        """Test position auto-correction to make area fit to field."""
        bsca = GameOfLife(TestExperiment)
        vals = {'state': RandInt(0, 1)}
        bb = BigBang(pos=(90, 90), size=(20, 20), vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        bb.generate(cells, bsca)
        self.assertEqual(bb.pos[0], 80, "Wrong X position.")
        self.assertEqual(bb.pos[1], 80, "Wrong Y position.")


class TestPrimordialSoup(TestPatternBase):
    """Tests for ``PrimordialSoup`` class."""

    def test_2d(self):
        """Test same field is generated with same seed."""
        bsca = GameOfLife(TestExperiment)
        vals = {'state': RandInt(0, 1)}
        seed = PrimordialSoup(vals=vals)
        cells = np.zeros((10000, ), dtype=np.int32)
        seed.random = LocalRandom("test")
        seed.generate(cells, bsca)
        self.assertEqual(binascii.crc32(cells[:10000]), 2251764292,
                         "Wrong field checksum.")
