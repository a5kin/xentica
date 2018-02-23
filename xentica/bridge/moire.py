"""
Module with the bridge to `Moire`_ UI.

.. _Moire: https://github.com/a5kin/moire

"""
from xentica.bridge.base import Bridge

__all__ = ['MoireBridge', ]


class MoireBridge:
    """
    Class incaplulating the actions for Moire UI.

    ``[`` Speed simulation down.

    ``]`` Speed simulation up.

    ``SPACEBAR`` Pause/unpause simulation.

    ``F12`` Toggle system info.

    ``ESC`` Exit app.

    """

    key_actions = {
        "[": Bridge.speed(-1),
        "]": Bridge.speed(1),
        "spacebar": Bridge.toggle_pause,
        "f12": Bridge.toggle_sysinfo,
        "escape": Bridge.exit_app,
    }
