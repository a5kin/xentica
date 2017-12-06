"""Tests for ``xentica.core.mixins`` module."""
import unittest

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.exceptions import XenticaException


class TestBscaDetector(unittest.TestCase):
    """Tests for ``BscaDetectorMixin`` class."""

    def test_not_detected(self):
        """Test exception is raised if BSCA not detected."""
        class Dummy(BscaDetectorMixin):
            pass
        with self.assertRaises(XenticaException):
            dummy = Dummy()
            dummy._bsca
