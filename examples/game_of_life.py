from hecate import core
from hecate.init import patterns
import moire


class GameOfLife(core.CellularAutomaton):
    """ The Idea of classic CA built with HECATE framework """
    state = core.IntegerProperty(max_val=1)

    class Topology:
        lattice = core.OrthogonalLattice(dimensions=2)
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    def emit(self):
        for i in range(len(self.buffers)):
            self.buffers_out[i] = self.state

    def absorb(self):
        neighbors_alive = core.IntegerVariable()
        for i in range(len(self.buffers)):
            neighbors_alive += self.buffers_in[i]
        is_born = (8 >> neighbors_alive) & 1
        is_sustain = (12 >> neighbors_alive) & 1
        self.state = is_born | is_sustain


class GOLExperiment(core.Experiment):
    """ Particular experiment, to be loaded at runtime in future """
    seed = "HECATE FIRST EXPERIMENT"
    dim = (960, 540)
    init = patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "state": patterns.RandInt(0, 1),
        }
    )


if __name__ == "__main__":
    ca = GameOfLife(GOLExperiment)
    gui = moire.GUI(runnable=ca)
    gui.run()
