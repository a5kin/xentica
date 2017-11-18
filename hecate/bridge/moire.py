from hecate.bridge.base import Bridge


class MoireBridge:

    key_actions = {
        "up": Bridge.move(0, 1),
        "down": Bridge.move(0, -1),
        "left": Bridge.move(-1, 0),
        "right": Bridge.move(1, 0),
        "=": Bridge.zoom(1),
        "-": Bridge.zoom(-1),
        "[": Bridge.speed(-1),
        "]": Bridge.speed(1),
        "spacebar": Bridge.toggle_pause,
        "f12": Bridge.toggle_sysinfo,
        "escape": Bridge.exit_app,
    }
