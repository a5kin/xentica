"""Tests for ``xentica.core.base`` module."""
import unittest
import os
import binascii

import numpy as np

from xentica.core.base import CellularAutomaton
from xentica.core.exceptions import XenticaException
from xentica.core.properties import IntegerProperty
from examples.game_of_life import (
    GameOfLife, GameOfLifeStatic, GameOfLifeColor, GameOfLife6D,
    GOLExperiment, GOLExperiment2, GOLExperimentColor,
    LifelikeCA, DiamoebaExperiment,
)
from examples.shifting_sands import ShiftingSands, ShiftingSandsExperiment
from examples.evolife import (
    EvoLife, CrossbreedingExperiment, CrossbreedingExperiment2
)


class TestCellularAutomaton(unittest.TestCase):
    """Tests for ``CellularAutomaton`` class."""

    num_steps = 1000
    num_runs = 3

    def test_single_ca(self):
        """
        Test single CA model.

        Run vanilla GoL for 1000 steps and make sure board checksum is correct.

        """
        for _ in range(self.num_runs):
            model = GameOfLife(GOLExperiment)
            for _ in range(self.num_steps):
                model.step()
            cells = model.gpu.arrays.cells.get()[:model.cells_num]
            checksum = binascii.crc32(cells)
            self.assertEqual(2981695958, checksum, "Wrong field checksum.")

    def test_multiple_ca(self):
        """Test two CellularAutomaton instances could be ran in parallel."""
        mod1 = GameOfLife(GOLExperiment)
        mod2 = GameOfLife(GOLExperiment)
        for _ in range(self.num_steps):
            mod1.step()
            mod2.step()
        checksum = binascii.crc32(mod1.gpu.arrays.cells.get()[:mod1.cells_num])
        self.assertEqual(2981695958, checksum, "Wrong field checksum (CA #1).")
        checksum = binascii.crc32(mod2.gpu.arrays.cells.get()[:mod2.cells_num])
        self.assertEqual(2981695958, checksum, "Wrong field checksum (CA #2).")

    def test_render(self):
        """
        Test basic rendering working.

        Run vanilla GoL for 1000 steps and check resulting image's checksum.

        """
        experiment = GOLExperiment
        experiment.zoom = 1
        model = GameOfLife(experiment)
        model.set_viewport(experiment.size)
        for _ in range(self.num_steps):
            model.step()
        img = model.render()
        self.assertEqual(1955702083, binascii.crc32(img / 255),
                         "Wrong image checksum.")

    def test_pause(self):
        """Test CA is not evolving when paused."""
        model = GameOfLife(GOLExperiment)
        model.paused = False
        cells_num = model.cells_num
        cells = model.gpu.arrays.cells.get()[:cells_num]
        checksum_before = binascii.crc32(cells)
        model.step()
        cells = model.gpu.arrays.cells.get()[:cells_num]
        checksum_after = binascii.crc32(cells)
        self.assertNotEqual(checksum_before, checksum_after,
                            "CA is paused.")
        model.paused = True
        checksum_before = checksum_after
        model.step()
        cells = model.gpu.arrays.cells.get()[:cells_num]
        checksum_after = binascii.crc32(cells)
        self.assertEqual(checksum_before, checksum_after,
                         "CA is not paused.")

    def test_save_load(self):
        """Save CA and test it's state is identical after load."""
        ca1 = GameOfLife(GOLExperiment)
        for _ in range(self.num_steps // 2):
            ca1.step()
        ca1.save("test.ca")
        ca2 = GameOfLife(GOLExperiment)
        ca2.load("test.ca")
        for _ in range(self.num_steps // 2):
            ca2.step()
        checksum = binascii.crc32(ca2.gpu.arrays.cells.get()[:ca2.cells_num])
        self.assertEqual(2981695958, checksum, "Wrong field checksum.")
        os.remove("test.ca")

    def test_load_random(self):
        """Save CA and test it's RNG state is identical after load."""
        ca1 = GameOfLife(GOLExperiment)
        ca1.save("test.ca")
        ca2 = GameOfLife(GOLExperiment)
        ca2.load("test.ca")
        self.assertEqual(ca1.random.standard.randint(1, 1000),
                         ca2.random.standard.randint(1, 1000),
                         "Wrong standard RNG state.")
        self.assertEqual(ca1.random.numpy.randint(1, 1000),
                         ca2.random.numpy.randint(1, 1000),
                         "Wrong numpy RNG state.")
        os.remove("test.ca")

    def test_no_topology(self):
        """Test exception is raised if ``Topology`` is not declared."""
        with self.assertRaises(XenticaException):
            class NoTopologyCA(CellularAutomaton):
                """CA without topology declared, to test exceptions."""
            self.assertEqual(type(NoTopologyCA), "",
                             "This line never should be executed.")

    def test_empty_topology(self):
        """Test exception is raised if ``Topology`` is empty."""
        with self.assertRaises(XenticaException):
            class NoLatticeCA(CellularAutomaton):
                """CA with empty topology to test exceptions."""
                class Topology:
                    """Empty topology."""
            self.assertEqual(type(NoLatticeCA), "",
                             "This line never should be executed.")

    def test_static_border(self):
        """Test exception is raised if ``Topology`` is empty."""
        model = GameOfLifeStatic(GOLExperiment)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertNotEqual(2981695958, checksum,
                            "Checksum shoud be different from parent class.")
        self.assertEqual(1098273940, checksum, "Wrong field checksum.")

    def test_multiple_properties(self):
        """Test CA with multiple properties works correctly."""
        model = GameOfLifeColor(GOLExperimentColor)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(532957133, checksum, "Wrong field checksum.")

    def test_multidimensional(self):
        """Test 6-dimensional CA works correctly."""
        class GOLExperiment6DLite(GOLExperiment2):
            """Small 6D board to test higher dimensions."""
            size = (64, 36, 3, 3, 3, 3)
        model = GameOfLife6D(GOLExperiment6DLite)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(2742543959, checksum, "Wrong field checksum.")

    def test_cell_width(self):
        """Test CA with 16 bit/cell works correctly."""
        class GameOfLifeInt(GameOfLife):
            """16-bit model to test bit width."""
            state = IntegerProperty(max_val=2 ** 16 - 1)
        model = GameOfLifeInt(GOLExperiment)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num].astype(np.uint8)
        checksum = binascii.crc32(cells)
        self.assertEqual(2981695958, checksum, "Wrong field checksum.")

    def test_nonuniform_interactions(self):
        """Test non-uniform buffer interactions."""
        model = ShiftingSands(ShiftingSandsExperiment)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(1117367015, checksum, "Wrong field checksum.")

    def test_unset_viewport(self):
        """Test correct exception is raised when viewport is not set."""
        with self.assertRaises(XenticaException):
            model = ShiftingSands(ShiftingSandsExperiment)
            model.render()

    def test_genetics_general(self):
        """Test general genetics stuff."""
        model = EvoLife(CrossbreedingExperiment, legacy_coloring=True)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(3351683720, checksum, "Wrong field checksum.")
        model = EvoLife(CrossbreedingExperiment2)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(3791001013, checksum, "Wrong field checksum.")

    def test_interactive(self):
        """Test CA with interactive parameter works correctly."""
        model = LifelikeCA(DiamoebaExperiment)
        for _ in range(self.num_steps):
            model.step()
        cells = model.gpu.arrays.cells.get()[:model.cells_num]
        checksum = binascii.crc32(cells)
        self.assertEqual(4016906756, checksum, "Wrong field checksum.")
