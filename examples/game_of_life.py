"""
A collection of models derived from Conway's Game Of Life.

Experiment classes included.

"""
from xentica import core
from xentica import seeds
from xentica.core import color_effects


class GameOfLife(core.CellularAutomaton):
    """
    The classic CA built with Xentica framework.

    It has only one property called ``state``, which is positive
    integer with max value of 1.

    """

    state = core.IntegerProperty(max_val=1)

    class Topology:
        """
        Mandatory class for all ``CellularAutomaton`` instances.

        All class variables below are also mandatory.

        Here, we declare the topology as a 2-dimensional orthogonal
        lattice with Moore neighborhood, wrapped to a 3-torus.

        """

        dimensions = 2
        lattice = core.OrthogonalLattice()
        neighborhood = core.MooreNeighborhood()
        border = core.TorusBorder()

    def emit(self):
        """
        Implement the logic of emit phase.

        Statements below will be translated into C code as emit kernel
        at the moment of class creation.

        Here, we just copy main state to surrounding buffers.

        """
        for i in range(len(self.buffers)):
            self.buffers[i].state = self.main.state

    def absorb(self):
        """
        Implement the logic of absorb phase.

        Statements below will be translated into C code as well.

        Here, we sum all neigbors buffered states and apply Conway
        rule to modify cell's own state.

        """
        neighbors_alive = core.IntegerVariable()
        for i in range(len(self.buffers)):
            neighbors_alive += self.neighbors[i].buffer.state
        is_born = (8 >> neighbors_alive) & 1
        is_sustain = (12 >> neighbors_alive) & 1
        self.main.state = is_born | is_sustain & self.main.state

    @color_effects.MovingAverage
    def color(self):
        """
        Implement the logic of cell's color calculation.

        Must return a tuple of RGB values computed from ``self.main``
        properties.

        Also, must be decorated by a class from ``color_effects``
        module.

        Here, we simply define 0 state as pure black, and 1 state as
        pure white.

        """
        r = self.main.state * 255
        g = self.main.state * 255
        b = self.main.state * 255
        return (r, g, b)


class GameOfLifeStatic(GameOfLife):
    """
    Game of Life variant with static border made of live cells.

    This is an example of how easy you can inherit other models.

    """

    class Topology(GameOfLife.Topology):
        """
        You can inherit parent class ``Topology``.

        Then, override only necessary variables.

        """

        border = core.StaticBorder(1)


class GameOfLifeColor(GameOfLife):
    """ Same Game Of Life, but with color per each cell """
    state = core.IntegerProperty(max_val=1)
    red = core.IntegerProperty(max_val=255)
    green = core.IntegerProperty(max_val=255)
    blue = core.IntegerProperty(max_val=255)

    def emit(self):
        for i in range(len(self.buffers)):
            self.buffers[i].state = self.main.state
            self.buffers[i].red = self.main.red
            self.buffers[i].green = self.main.green
            self.buffers[i].blue = self.main.blue

    def absorb(self):
        GameOfLife.absorb(self)
        red_sum = core.IntegerVariable()
        for i in range(len(self.buffers)):
            red_sum += self.neighbors[i].buffer.red + 1
        green_sum = core.IntegerVariable()
        for i in range(len(self.buffers)):
            green_sum += self.neighbors[i].buffer.green + 1
        blue_sum = core.IntegerVariable()
        for i in range(len(self.buffers)):
            blue_sum += self.neighbors[i].buffer.blue + 1
        self.main.red = red_sum * self.main.state
        self.main.green = green_sum * self.main.state
        self.main.blue = blue_sum * self.main.state

    @color_effects.MovingAverage
    def color(self):
        r = self.main.state * self.main.red
        g = self.main.state * self.main.green
        b = self.main.state * self.main.blue
        return (r, g, b)


class GameOfLife6D(GameOfLife):
    """
    Conway rules in 6D.
    That's really all you need to go into hyperspace :)

    """
    class Topology(GameOfLife.Topology):
        dimensions = 6
        neighborhood = core.VonNeumannNeighborhood()


class GOLExperiment(core.Experiment):
    """ Particular experiment, to be loaded at runtime in future """
    word = "OBEY XENTICA"
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
    word = "XENTICA IS YOUR GODDESS"
    size = (640, 360, )
    zoom = 3
    pos = [0, 0]
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


class GOLExperimentColor(GOLExperiment):
    """ Special case for GameOfLifeColor """
    fade_out = 10
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
            "red": seeds.random.RandInt(0, 255),
            "green": seeds.random.RandInt(0, 255),
            "blue": seeds.random.RandInt(0, 255),
        }
    )


class GOLExperiment6D(GOLExperiment2):
    """ Special case for 6D Life """
    size = (640, 360, 3, 3, 3, 3)


if __name__ == "__main__":
    import moire
    ca = GameOfLifeColor(GOLExperimentColor)
    gui = moire.GUI(runnable=ca)
    gui.run()
