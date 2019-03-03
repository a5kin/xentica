"""
Factitious CA to test non-uniform buffer interactions.

Experiment classes included.

"""
from xentica import core
from xentica import seeds
from xentica.core import color_effects

from .base import RegularCA, RegularExperiment
from .base import run_simulation


class ShiftingSands(RegularCA):
    """
    CA for non-uniform buffer interactions test.

    It emits the whole value to a constant direction, then absorbs
    surrounding values by summing them.

    """
    state = core.IntegerProperty(max_val=1)

    def emit(self):
        """Emit the whole value to a constant direction."""
        direction = 0
        for i in range(len(self.buffers)):
            if i == direction:
                self.buffers[i].state = self.main.state
            else:
                self.buffers[i].state = 0

    def absorb(self):
        """Absorb surrounding values by summing them."""
        new_val = core.IntegerVariable()
        for i in range(len(self.buffers)):
            new_val += self.neighbors[i].buffer.state
        self.main.state = new_val

    @color_effects.MovingAverage
    def color(self):
        """Render contrast black & white cells."""
        red = self.main.state * 255
        green = self.main.state * 255
        blue = self.main.state * 255
        return (red, green, blue)


class ShiftingSandsExperiment(RegularExperiment):
    """Vanilla experiment with randomly initialized area."""

    word = "LET IT SHIFT"
    seed = seeds.patterns.BigBang(
        pos=(320, 180),
        size=(100, 100),
        vals={
            "state": seeds.random.RandInt(0, 1),
        }
    )


if __name__ == "__main__":
    run_simulation(ShiftingSands, ShiftingSandsExperiment)
