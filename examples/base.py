"""Base classes and helpers commonly used in examples."""

import abc

from xentica import core


class RegularCA(core.CellularAutomaton):
    """
    Template for the most regular CA topology.

    Does not implements any logic.

    """

    class Topology:
        """2D Moore neighborhood, wrapped to a 3-torus."""

        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    @abc.abstractmethod
    def emit(self):
        """Emit phase logic."""

    @abc.abstractmethod
    def absorb(self):
        """Absorb phase logic."""

    @abc.abstractmethod
    def color(self):
        """Coloring logic."""


class RegularExperiment(core.Experiment):
    """Experiment with the most common field size, zoom and pos."""

    word = "I AM MUNDANE"
    size = (640, 360, )
    zoom = 3
    pos = [0, 0]


def run_simulation(model, experiment):
    """Run model/experiment interactively."""
    import moire
    mod = model(experiment)
    gui = moire.GUI(runnable=mod)
    gui.run()
