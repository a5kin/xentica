import unittest

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
            self.assertEqual(1172, np.sum(ca.cells_gpu.get()),
                             "Wrong field checksum.")

    def test_multiple_ca(self):
        ca1 = GameOfLife(GOLExperiment)
        ca2 = GameOfLife(GOLExperiment)
        for j in range(self.num_steps):
            ca1.step()
            ca2.step()
        self.assertEqual(1172, np.sum(ca1.cells_gpu.get()),
                         "Wrong field checksum (CA #1).")
        self.assertEqual(1172, np.sum(ca2.cells_gpu.get()),
                         "Wrong field checksum (CA #2).")
