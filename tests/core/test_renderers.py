"""Tests for ``xentica.core.renderers`` module."""
import unittest

from xentica.core.renderers import Renderer


class TestRenderer(unittest.TestCase):
    """Tests for ``Renderer`` class."""

    def test_base(self):
        """Test base class is returning empty code for kernel."""
        renderer = Renderer()
        code = renderer.render_code()
        self.assertEqual(code, "", "Base class kernel should be empty")
