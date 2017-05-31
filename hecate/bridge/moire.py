from hecate.bridge.base import Bridge


class MoireBridge:

    key_actions = {
        "up": Bridge.move_up,
        "down": Bridge.move_down,
        "left": Bridge.move_left,
        "right": Bridge.move_right,
        "=": Bridge.zoom_in,
        "-": Bridge.zoom_out,
        "[": Bridge.speed_down,
        "]": Bridge.speed_up,
        "escape": Bridge.exit_app,
    }
