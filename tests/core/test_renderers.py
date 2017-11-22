import unittest

from xentica.core.renderers import Renderer


class TestRenderer(unittest.TestCase):

    def test_base(self):
        renderer = Renderer()
        code = renderer.render_code()
        self.assertEqual(code, "", "Base class kernel should be empty")
