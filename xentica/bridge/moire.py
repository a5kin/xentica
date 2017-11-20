from xentica.bridge.base import Bridge


class MoireBridge:

    key_actions = {
        "[": Bridge.speed(-1),
        "]": Bridge.speed(1),
        "spacebar": Bridge.toggle_pause,
        "f12": Bridge.toggle_sysinfo,
        "escape": Bridge.exit_app,
    }
