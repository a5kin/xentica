import unittest

from hecate.bridge.base import Bridge
from examples.game_of_life import GameOfLife, GOLExperiment


class SysInfo:

    def toggle(self):
        pass


class DummyGui:

    def __init__(self):
        self.sysinfo = SysInfo()

    def exit_app(self):
        pass


class TestBridge(unittest.TestCase):

    def setUp(self):
        self.ca = GameOfLife(GOLExperiment)
        self.gui = DummyGui()

    def test_noop(self):
        Bridge.noop(self.ca, self.gui)

    def test_exit(self):
        Bridge.exit_app(self.ca, self.gui)

    def test_move_up(self):
        y = self.ca.pos[1]
        move_up = self.ca.bridge.key_actions['up']
        move_up(self.ca, self.gui)
        self.assertEqual((y + 1) % self.ca.size[1], self.ca.pos[1])

    def test_move_down(self):
        y = self.ca.pos[1]
        move_down = self.ca.bridge.key_actions['down']
        move_down(self.ca, self.gui)
        self.assertEqual((y - 1) % self.ca.size[1], self.ca.pos[1])

    def test_move_right(self):
        x = self.ca.pos[0]
        move_right = self.ca.bridge.key_actions['right']
        move_right(self.ca, self.gui)
        self.assertEqual((x + 1) % self.ca.size[0], self.ca.pos[0])

    def test_move_left(self):
        x = self.ca.pos[0]
        move_left = self.ca.bridge.key_actions['left']
        move_left(self.ca, self.gui)
        self.assertEqual((x - 1) % self.ca.size[0], self.ca.pos[0])

    def test_zoom_in(self):
        zoom = self.ca.zoom
        zoom_in = self.ca.bridge.key_actions['=']
        zoom_in(self.ca, self.gui)
        self.assertEqual(zoom + 1, self.ca.zoom)

    def test_zoom_out(self):
        zoom = self.ca.zoom
        zoom_out = self.ca.bridge.key_actions['-']
        zoom_out(self.ca, self.gui)
        self.assertEqual(max(zoom - 1, 1), self.ca.zoom)

    def test_speed_up(self):
        speed = self.ca.speed
        speed_up = self.ca.bridge.key_actions[']']
        speed_up(self.ca, self.gui)
        self.assertEqual(speed + 1, self.ca.speed)

    def test_speed_down(self):
        speed = self.ca.speed
        speed_down = self.ca.bridge.key_actions['[']
        speed_down(self.ca, self.gui)
        self.assertEqual(max(speed - 1, 1), self.ca.speed)

    def test_toggle_pause(self):
        paused = self.ca.paused
        Bridge.toggle_pause(self.ca, self.gui)
        self.assertEqual(not paused, self.ca.paused)

    def test_toggle_sysinfo(self):
        Bridge.toggle_sysinfo(self.ca, self.gui)
