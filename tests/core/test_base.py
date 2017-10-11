import unittest
import os
import binascii

from hecate.core.base import HecateException, CellularAutomaton
from examples.game_of_life import (
    GameOfLife, GameOfLifeStatic,
    GOLExperiment,
)


class TestCellularAutomaton(unittest.TestCase):

    num_steps = 1000
    num_runs = 3

    def test_single_ca(self):
        for i in range(self.num_runs):
            ca = GameOfLife(GOLExperiment)
            for j in range(self.num_steps):
                ca.step()
            checksum = binascii.crc32(ca.cells_gpu.get()[:ca.cells_num])
            self.assertEqual(3395585361, checksum, "Wrong field checksum.")

    def test_multiple_ca(self):
        ca1 = GameOfLife(GOLExperiment)
        ca2 = GameOfLife(GOLExperiment)
        for j in range(self.num_steps):
            ca1.step()
            ca2.step()
        checksum = binascii.crc32(ca1.cells_gpu.get()[:ca1.cells_num])
        self.assertEqual(3395585361, checksum, "Wrong field checksum (CA #1).")
        checksum = binascii.crc32(ca2.cells_gpu.get()[:ca2.cells_num])
        self.assertEqual(3395585361, checksum, "Wrong field checksum (CA #2).")

    def test_render(self):
        experiment = GOLExperiment
        experiment.zoom = 1
        ca = GameOfLife(experiment)
        ca.set_viewport(experiment.size)
        for j in range(self.num_steps):
            ca.step()
        img = ca.render()
        self.assertEqual(4150286101, binascii.crc32(img / 255),
                         "Wrong image checksum.")

    def test_pause(self):
        ca = GameOfLife(GOLExperiment)
        ca.paused = False
        checksum_before = binascii.crc32(ca.cells_gpu.get()[:ca.cells_num])
        ca.step()
        checksum_after = binascii.crc32(ca.cells_gpu.get()[:ca.cells_num])
        self.assertNotEqual(checksum_before, checksum_after,
                            "CA is paused.")
        ca.paused = True
        checksum_before = checksum_after
        ca.step()
        checksum_after = binascii.crc32(ca.cells_gpu.get()[:ca.cells_num])
        self.assertEqual(checksum_before, checksum_after,
                         "CA is not paused.")

    def test_save_load(self):
        ca1 = GameOfLife(GOLExperiment)
        for i in range(self.num_steps // 2):
            ca1.step()
        ca1.save("test.ca")
        ca2 = GameOfLife(GOLExperiment)
        ca2.load("test.ca")
        for i in range(self.num_steps // 2):
            ca2.step()
        checksum = binascii.crc32(ca2.cells_gpu.get()[:ca2.cells_num])
        self.assertEqual(3395585361, checksum, "Wrong field checksum.")
        os.remove("test.ca")

    def test_load_random(self):
        ca1 = GameOfLife(GOLExperiment)
        ca1.save("test.ca")
        ca2 = GameOfLife(GOLExperiment)
        ca2.load("test.ca")
        self.assertEqual(ca1.random.std.randint(1, 1000),
                         ca2.random.std.randint(1, 1000),
                         "Wrong standard RNG state.")
        self.assertEqual(ca1.random.np.randint(1, 1000),
                         ca2.random.np.randint(1, 1000),
                         "Wrong numpy RNG state.")
        os.remove("test.ca")

    def test_no_topology(self):
        with self.assertRaises(HecateException):
            class NoTopologyCA(CellularAutomaton):
                pass

    def test_empty_topology(self):
        with self.assertRaises(HecateException):
            class NoLatticeCA(CellularAutomaton):
                class Topology:
                    pass

    def test_static_border(self):
        ca = GameOfLifeStatic(GOLExperiment)
        for j in range(self.num_steps):
            ca.step()
        checksum = binascii.crc32(ca.cells_gpu.get()[:ca.cells_num])
        self.assertNotEqual(3395585361, checksum,
                            "Checksum shoud be different from parent class.")
        self.assertEqual(2543376250, checksum, "Wrong field checksum.")
