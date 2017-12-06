"""Tests for ``xentica.core.topology.neighborhood`` module."""
import unittest

from xentica.core.exceptions import XenticaException
from xentica.core.topology.neighborhood import (
    MooreNeighborhood,
)


class TestMooreNeighborhood(unittest.TestCase):
    """Tests for ``MooreNeighborhood`` class."""

    def test_incorrect_dimensions(self):
        """Test exception is raised for incorrect dimensionality."""
        neighborhood = MooreNeighborhood()
        with self.assertRaises(XenticaException):
            neighborhood.dimensions = 0
