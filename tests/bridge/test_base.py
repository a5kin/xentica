"""Tests for ``xentica.bridge.base`` module."""
import unittest

from xentica.bridge.base import Bridge
from examples.game_of_life import GameOfLife, GOLExperiment


class SysInfo:
    """Class emulating ``SysInfo`` interface."""

    def toggle(self):
        """Do nothing on toggle."""


class DummyGui:
    """Class emulating basic GUI."""

    def __init__(self):
        """Initialize ``SysInfo``."""
        self.sysinfo = SysInfo()

    def exit_app(self):
        """Do nothing on exit."""


class TestBridge(unittest.TestCase):
    """Tests for ``Bribge`` class."""

    def setUp(self):
        """Set up necessary things before each test."""
        self.model = GameOfLife(GOLExperiment)
        self.gui = DummyGui()

    def test_noop(self):
        """Test no-op action."""
        Bridge.noop(self.model, self.gui)

    def test_exit(self):
        """Test exit action."""
        Bridge.exit_app(self.model, self.gui)

    def test_move_up(self):
        """Test move up action."""
        coord_y = self.model.pos[1]
        move_up = self.model.bridge.key_actions['up']
        move_up(self.model, self.gui)
        self.assertEqual((coord_y + 1) % self.model.size[1], self.model.pos[1])

    def test_move_down(self):
        """Test move down action."""
        coord_y = self.model.pos[1]
        move_down = self.model.bridge.key_actions['down']
        move_down(self.model, self.gui)
        self.assertEqual((coord_y - 1) % self.model.size[1], self.model.pos[1])

    def test_move_right(self):
        """Test move right action."""
        coord_x = self.model.pos[0]
        move_right = self.model.bridge.key_actions['right']
        move_right(self.model, self.gui)
        self.assertEqual((coord_x + 1) % self.model.size[0], self.model.pos[0])

    def test_move_left(self):
        """Test move left action."""
        coord_x = self.model.pos[0]
        move_left = self.model.bridge.key_actions['left']
        move_left(self.model, self.gui)
        self.assertEqual((coord_x - 1) % self.model.size[0], self.model.pos[0])

    def test_zoom_in(self):
        """Test zoom in action."""
        zoom = self.model.zoom
        zoom_in = self.model.bridge.key_actions['=']
        zoom_in(self.model, self.gui)
        self.assertEqual(zoom + 1, self.model.zoom)

    def test_zoom_out(self):
        """Test zoom out action."""
        zoom = self.model.zoom
        zoom_out = self.model.bridge.key_actions['-']
        zoom_out(self.model, self.gui)
        self.assertEqual(max(zoom - 1, 1), self.model.zoom)

    def test_speed_up(self):
        """Test speed up action."""
        speed = self.model.speed
        speed_up = self.model.bridge.key_actions[']']
        speed_up(self.model, self.gui)
        self.assertEqual(speed + 1, self.model.speed)

    def test_speed_down(self):
        """Test speed down action."""
        speed = self.model.speed
        speed_down = self.model.bridge.key_actions['[']
        speed_down(self.model, self.gui)
        self.assertEqual(max(speed - 1, 1), self.model.speed)

    def test_toggle_pause(self):
        """Test toggle pause action."""
        paused = self.model.paused
        Bridge.toggle_pause(self.model, self.gui)
        self.assertEqual(not paused, self.model.paused)

    def test_toggle_sysinfo(self):
        """Test toggle sysinfo action."""
        Bridge.toggle_sysinfo(self.model, self.gui)
