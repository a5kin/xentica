"""Tests for ``xentica.core.topology.lattice`` module."""
import unittest

from xentica.core.exceptions import XenticaException
from xentica.core.topology.lattice import (
    OrthogonalLattice,
)
from examples.game_of_life import GameOfLife, GOLExperiment


class GOLExperimentBroken(GOLExperiment):
    """Experiment with broken field size."""
    size = (23, )


class TestOrthogonalLattice(unittest.TestCase):
    """Tests for ``OrthogonalLattice`` class."""

    def test_incorrect_dimensions(self):
        """Test exception is raised for incorrect dimensionality."""
        lattice = OrthogonalLattice()
        with self.assertRaises(XenticaException):
            lattice.dimensions = 0

    def test_wrong_field_size(self):
        """Test wrong field size set in experiment."""
        with self.assertRaises(XenticaException):
            GameOfLife(GOLExperimentBroken)
