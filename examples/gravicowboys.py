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
    energy = core.IntegerProperty(max_val=2 ** 12 - 1)
    new_energy = core.IntegerProperty(max_val=2 ** 12 - 1)
    gravity = core.FloatProperty()

    def emit(self):
        """Emit the energy to field's gradient direction."""
        self.main.new_energy = 1 * self.main.energy
        max_gravity = core.FloatVariable()
        max_gravity += xmath.min(*[nbr.main.gravity for nbr in self.neighbors])
        energy_passed = core.IntegerVariable()
        energy_passed *= 0
        denergy = core.IntegerVariable()
        for i, buf in enumerate(self.buffers):
            fit_dir = max_gravity == self.neighbors[i].main.gravity
            has_energy = (self.main.energy - energy_passed) > 0
            denergy *= 0
            denergy += fit_dir * has_energy * 1
            energy_passed += denergy
            buf.energy = denergy
            buf.gravity = self.main.gravity + self.main.energy
        # energy_passed = 0
        self.main.new_energy -= energy_passed

    def absorb(self):
        """Absorb incoming masses and spread gravity."""
        energy_passed = core.IntegerVariable()
        for i in range(len(self.buffers)):
            energy_passed += self.neighbors[i].buffer.energy
        self.main.energy = self.main.new_energy + energy_passed

        new_gravity = core.FloatVariable()
        for i in range(len(self.buffers)):
            new_gravity += self.neighbors[i].buffer.gravity
        self.main.gravity = new_gravity / (len(self.buffers) * 1.001)

    @color_effects.MovingAverage
    def color(self):
        """Render mass as yellow, gravity as blue."""
        red = self.main.energy * 255
        green = xmath.int(self.main.gravity * 108)
        blue = xmath.int(self.main.gravity * 108)
        return (red, green, blue)


class GraviCowboysExperiment(RegularExperiment):
    """Vanilla experiment with randomly initialized area."""

    word = "LET IT GROOVE"
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(10, 10),
        vals={
            "energy": seeds.random.RandInt(0, 1) * 1,
            "gravity": 0,
        }
    )
    seed += seeds.patterns.BigBang(
        pos=(20, 80),
        size=(10, 10),
        vals={
            "energy": seeds.random.RandInt(0, 1) * 1,
            "gravity": 0,
        }
    )
    seed += seeds.patterns.BigBang(
        pos=(400, 80),
        size=(10, 10),
        vals={
            "energy": seeds.random.RandInt(0, 1) * 1,
            "gravity": 0,
        }
    )


if __name__ == "__main__":
    run_simulation(GraviCowboys, GraviCowboysExperiment)
