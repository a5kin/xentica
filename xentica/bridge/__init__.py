"""
The bridge between Xentica and GUI interface.

This package contains all necessary stuff to connect Xentica framework
to custom interactive visualization environments.

Right now, only one environment (`Moire`_) is available. This is the
official environment, evolving along with the main framework. You are
free to implement your own environments. If so, please make a PR on
Github and we'll include your solution to the bridge.

Bridge functions are automatically used when you run the simulation
like this::

    import moire
    ca = MyCellularAutomaton(MyExperiment)
    gui = moire.GUI(runnable=ca)
    gui.run()

.. _Moire: https://github.com/a5kin/moire

"""
from xentica.bridge.moire import MoireBridge


__all__ = [
    "MoireBridge",
]
