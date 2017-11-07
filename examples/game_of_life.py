from hecate import core
from hecate import seeds
from hecate.core import color_effects


class GameOfLife(core.CellularAutomaton):
    """ The Idea of classic CA built with HECATE framework """
    state = core.IntegerProperty(max_val=1)

    class Topology:
        dimensions = 2
        lattice = core.OrthogonalLattice()
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
        self.main.state = is_born | is_sustain & self.main.state

    @color_effects.MovingAverage
    def color(self):
        r = self.main.state * 255
        g = self.main.state * 255
        b = self.main.state * 255
        return (r, g, b)


class GameOfLifeStatic(GameOfLife):
    class Topology:
        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.StaticBorder(1)


class GOLExperiment(core.Experiment):
    """ Particular experiment, to be loaded at runtime in future """
    word = "HECATE FIRST EXPERIMENT"
    size = (640, 360, )
    zoom = 3
    pos = [0, 0]
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


class GOLExperiment2(core.Experiment):
    """ Experiment initialized with Primordial Soup pattern. """
    word = "HECATE FIRST EXPERIMENT"
    size = (640, 360, )
    zoom = 3
    pos = [0, 0]
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


if __name__ == "__main__":
    import moire
    ca = GameOfLife(GOLExperiment)
    gui = moire.GUI(runnable=ca)
    gui.run()
