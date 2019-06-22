"""
A collection of models derived from Conway's Game Of Life.

Experiment classes included.

"""
from xentica import core
from xentica import seeds
from xentica.core import color_effects
from xentica.tools.rules import LifeLike


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
        red = self.main.state * 255
        green = self.main.state * 255
        blue = self.main.state * 255
        return (red, green, blue)


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
    """
    Game Of Life variant with RGB color.

    This is an example of how to use multiple properties per cell.

    """

    state = core.IntegerProperty(max_val=1)
    red = core.IntegerProperty(max_val=255)
    green = core.IntegerProperty(max_val=255)
    blue = core.IntegerProperty(max_val=255)

    def emit(self):
        """Copy all properties to surrounding buffers."""
        for i in range(len(self.buffers)):
            self.buffers[i].state = self.main.state
            self.buffers[i].red = self.main.red
            self.buffers[i].green = self.main.green
            self.buffers[i].blue = self.main.blue

    def absorb(self):
        """
        Calculate RGB as neighbors sum for living cell only.

        Note, parent ``absorb`` method should be called using direct
        class access, not via ``super``.

        """
        GameOfLife.absorb(self)
        red_sum = core.IntegerVariable()
        green_sum = core.IntegerVariable()
        blue_sum = core.IntegerVariable()
        for i in range(len(self.buffers)):
            red_sum += self.neighbors[i].buffer.red + 1
            green_sum += self.neighbors[i].buffer.green + 1
            blue_sum += self.neighbors[i].buffer.blue + 1
        self.main.red = red_sum * self.main.state
        self.main.green = green_sum * self.main.state
        self.main.blue = blue_sum * self.main.state

    @color_effects.MovingAverage
    def color(self):
        """Calculate color as usual."""
        red = self.main.state * self.main.red
        green = self.main.state * self.main.green
        blue = self.main.state * self.main.blue
        return (red, green, blue)


class GameOfLife6D(GameOfLife):
    """
    Game of Life variant in 6D.

    Nothing interesting, just to prove you can do it with ease.

    """

    class Topology(GameOfLife.Topology):
        """
        Hyper-spacewalk, is as easy as increase ``dimensions`` value.

        However, we are also changing neighborhood to Von Neumann
        here, to prevent neighbors number exponential grow.

        """

        dimensions = 6
        neighborhood = core.VonNeumannNeighborhood()


class LifelikeCA(GameOfLife):
    """Lifelike CA with a flexible rule that could be changed at runtime."""
    rule = core.Parameter(
        default=LifeLike.golly2int("B3/S23"),
        interactive=True,
    )

    def absorb(self):
        """Implement parent's clone with a rule as a parameter."""
        neighbors_alive = core.IntegerVariable()
        for i in range(len(self.buffers)):
            neighbors_alive += self.neighbors[i].buffer.state
        is_born = (self.rule >> neighbors_alive) & 1
        is_sustain = (self.rule >> 9 >> neighbors_alive) & 1
        self.main.state = is_born | is_sustain & self.main.state

    def step(self):
        """Change the rule interactively after some time passed."""
        if self.timestep == 23:
            self.rule = LifeLike.golly2int("B3/S23")
        super(LifelikeCA, self).step()


class GOLExperiment(core.Experiment):
    """
    Particular experiment for the vanilla Game of Life.

    Here, we define constants and initial conditions from which the
    world's seed will be generated.

    The ``word`` is an RNG seed string. The ``size``, ``zoom`` and
    ``pos`` are board contstants. The ``seed`` is a pattern used in
    the initial board state generation.

    ``BigBang`` is a pattern when small area initialized with a
    high-density random values.

    """

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


class GOLExperiment2(GOLExperiment):
    """
    Another experiment for the vanilla GoL.

    Since it is inherited from ``GOLExperiment``, we can redefine only
    values we need.

    ``PrimordialSoup`` is a pattern when the whole board is
    initialized with low-density random values.

    """

    word = "XENTICA IS YOUR GODDESS"
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


class GOLExperimentColor(GOLExperiment):
    """
    The experiment for ``GameOfLifeColor``.

    Here, we introduce ``fade_out`` constant, which is used in
    rendering and slowly fading out the color of cells.

    Note, it is only an aestetic effect, and does not affect the real
    cell state.

    """

    fade_in = 255
    fade_out = 10
    smooth_factor = 1
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "state": seeds.random.RandInt(0, 1),
            "red": seeds.random.RandInt(0, 255),
            "green": seeds.random.RandInt(0, 255),
            "blue": seeds.random.RandInt(0, 255),
        }
    )


class GOLExperiment6D(GOLExperiment2):
    """
    Special experiment for 6D Life.

    Here, we define the world with 2 spatial and 4 looped
    micro-dimensions, 3 cells per micro-dimension.

    As a result, we get large quasi-stable oscillators, looping over
    micro-dimensions. Strangely formed, but nothing interesting,
    really.

    """

    size = (640, 360, 3, 3, 3, 3)


class DiamoebaExperiment(GOLExperiment):
    """Experiment with the interactive rule."""
    rule = LifeLike.golly2int("B35678/S5678")


def main():
    """Run model/experiment interactively."""
    import moire
    model = GameOfLifeColor(GOLExperimentColor)
    gui = moire.GUI(runnable=model)
    gui.run()


if __name__ == "__main__":
    main()
