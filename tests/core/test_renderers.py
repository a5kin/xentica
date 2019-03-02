"""Tests for ``xentica.core.renderers`` module."""
import unittest

from xentica.core.renderers import Renderer, RendererPlain


class TestRenderer(unittest.TestCase):
    """Tests for ``Renderer`` class."""

    def test_base(self):
        """Test base class is returning empty code for kernel."""
        renderer = Renderer()
        code = renderer.render_code()
        self.assertEqual(code, "", "Base class kernel should be empty")

    def test_renderer_plain(self):
        """Test ``RendererPlain`` class."""
        renderer = RendererPlain(projection_axes=(1, 0))
        self.assertEqual(renderer.projection_axes, (1, 0), "Wrong axis")
