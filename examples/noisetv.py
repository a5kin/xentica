"""Noise TV powered by Xentica."""

from xentica import core
from xentica import seeds
from xentica.core import color_effects
from xentica.seeds.random import RandInt
from xentica.tools import xmath

from examples.base import RegularCA, RegularExperiment
from examples.base import run_simulation


class NoiseTV(RegularCA):
    """Just pseudorandom noise, tickling your eyes."""
    rng = core.RandomProperty()

    def emit(self):
        """Do nothing, the noise is elementwise."""

    def absorb(self):
        """Make some noise... well not exactly."""
        return self.main.rng  # this will render next RNG value

    @color_effects.MovingAverage
    def color(self):
        """Make some noise, now for real."""
        val = xmath.int(self.main.rng.uniform * 255)
        return (val, val, val)


class NoiseTVExperiment(RegularExperiment):
    """Default experiment for legacy EvoLife."""

    word = "PSHSHSHSH!"
    seed = seeds.patterns.PrimordialSoup(
        vals={
            "rng": RandInt(0, 2 ** 16 - 1)
        }
    )


if __name__ == "__main__":
    run_simulation(NoiseTV, NoiseTVExperiment)
