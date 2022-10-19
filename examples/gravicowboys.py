"""
Simplified quantum gravitational field simulation.

This CA tests a hypothesis that global gravitational effects are
possible in CA using only local cells interactions.

"""


from xentica import core
from xentica import seeds
from xentica.core import color_effects
from xentica.tools import xmath

from examples.base import RegularCA, RegularExperiment
from examples.base import run_simulation


class GraviCowboys(RegularCA):
    """
    CA for gravitational field simulation.

    Each cell holds an integer energy (mass) level and a float
    gravitational field value. Next field value is calulated as a
    mean of all neighbors' masses + field values.  Then, cell is
    spreading an amount of energy into direction of the field's
    gradient.

    """
    energy = core.IntegerProperty(max_val=1)
    gravity = core.FloatProperty()

    def emit(self):
        """Emit the energy to field's gradient direction."""
        direction = -1  # TODO: find greatest field value
        for i, buf in enumerate(self.buffers):
            buf.gravity = self.main.gravity + self.main.energy
            if i == direction:
                buf.energy = self.main.energy
            else:
                buf.energy = 0

    def absorb(self):
        """Absorb incoming masses and spread gravity."""
        new_gravity = core.FloatVariable()
        for i in range(len(self.buffers)):
            new_gravity += self.neighbors[i].buffer.gravity
        self.main.gravity = new_gravity / (len(self.buffers) * 1)
        new_energy = core.IntegerVariable()
        # for i in range(len(self.buffers)):
        #     new_energy += self.neighbors[i].buffer.energy
        new_energy += 0 + self.main.energy
        self.main.energy = new_energy

    @color_effects.MovingAverage
    def color(self):
        """Render mass as yellow, gravity as blue."""
        red = self.main.energy * 255
        green = xmath.int(self.main.gravity * 255)
        blue = xmath.int(self.main.gravity * 255)
        return (red, green, blue)


class GraviCowboysExperiment(RegularExperiment):
    """Vanilla experiment with randomly initialized area."""

    word = "LET IT GROOVE"
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "energy": seeds.random.RandInt(0, 1),
            "gravity": 0,
        }
    )


if __name__ == "__main__":
    run_simulation(GraviCowboys, GraviCowboysExperiment)
