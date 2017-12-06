"""Tests for ``xentica.core.topology.border`` module."""
import unittest

from xentica.core.exceptions import XenticaException
from xentica.core.topology.border import (
    TorusBorder,
)


class TestTorusBorder(unittest.TestCase):
    """Tests for ``TorusBorder`` class."""

    def test_incorrect_dimensions(self):
        """Test exception is raised for incorrect dimensionality."""
        border = TorusBorder()
        with self.assertRaises(XenticaException):
            border.dimensions = 0
