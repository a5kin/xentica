import unittest
import os

import numpy as np

from examples.game_of_life import GameOfLife, GOLExperiment


class TestCellularAutomaton(unittest.TestCase):

    num_steps = 1000
    num_runs = 3

    def test_single_ca(self):
        for i in range(self.num_runs):
            ca = GameOfLife(GOLExperiment)
            for j in range(self.num_steps):
                ca.step()
            self.assertEqual(584, np.sum(ca.cells_gpu.get()[:ca.cells_num]),
                             "Wrong field checksum.")

    def test_multiple_ca(self):
        ca1 = GameOfLife(GOLExperiment)
        ca2 = GameOfLife(GOLExperiment)
        for j in range(self.num_steps):
            ca1.step()
            ca2.step()
        self.assertEqual(584, np.sum(ca1.cells_gpu.get()[:ca1.cells_num]),
                         "Wrong field checksum (CA #1).")
        self.assertEqual(584, np.sum(ca2.cells_gpu.get()[:ca2.cells_num]),
                         "Wrong field checksum (CA #2).")

    def test_render(self):
        experiment = GOLExperiment
        experiment.zoom = 1
        ca = GameOfLife(experiment)
        ca.set_viewport(experiment.size)
        for j in range(self.num_steps):
            ca.step()
        img = ca.render()
        self.assertEqual(584 * 3, np.sum(img / 255),
                         "Wrong image checksum.")

    def test_pause(self):
        ca = GameOfLife(GOLExperiment)
        ca.paused = False
        checksum_before = np.sum(ca.cells_gpu.get()[:ca.cells_num])
        ca.step()
        checksum_after = np.sum(ca.cells_gpu.get()[:ca.cells_num])
        self.assertNotEqual(checksum_before, checksum_after,
                            "CA is paused.")
        ca.paused = True
        checksum_before = checksum_after
        ca.step()
        checksum_after = np.sum(ca.cells_gpu.get()[:ca.cells_num])
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
        self.assertEqual(584, np.sum(ca2.cells_gpu.get()[:ca2.cells_num]),
                         "Wrong field checksum.")
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
