from hecate import core
from hecate import seeds
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
            self.buffers[i].state = self.main.state

    def absorb(self):
        neighbors_alive = core.IntegerVariable()
        for i in range(len(self.buffers)):
            neighbors_alive += self.neighbors[i].buffer.state
        is_born = (8 >> neighbors_alive) & 1
        is_sustain = (12 >> neighbors_alive) & 1
        self.main.state = is_born | is_sustain & self.main_state


class GOLExperiment(core.Experiment):
    """ Particular experiment, to be loaded at runtime in future """
    word = "HECATE FIRST EXPERIMENT"
    size = (1920, 1080, )
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


if __name__ == "__main__":
    ca = GameOfLife(GOLExperiment)
    gui = moire.GUI(runnable=ca)
    gui.run()
