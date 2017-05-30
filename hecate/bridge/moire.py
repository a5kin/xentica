from hecate.bridge.base import Bridge


class MoireBridge:

    key_actions = {
        "up": Bridge.noop,
        "down": Bridge.noop,
        "left": Bridge.noop,
        "right": Bridge.noop,
        "escape": Bridge.exit_app,
    }
