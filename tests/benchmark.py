"""Script to run Xentica benchmark."""
import time

from examples.game_of_life import (
    GameOfLife, GOLExperiment
)
from examples.shifting_sands import (
    ShiftingSands, ShiftingSandsExperiment
)
from examples.noisetv import (
    NoiseTV, NoiseTVExperiment,
)
from examples.evolife import (
    EvoLife, BigBangExperiment,
)
from xentica.utils.formatters import sizeof_fmt


MODELS = [
    ("Conway's Life", GameOfLife, GOLExperiment),
    ("Shifting Sands", ShiftingSands, ShiftingSandsExperiment),
    ("Noise TV", NoiseTV, NoiseTVExperiment),
    ("EvoLife", EvoLife, BigBangExperiment),
]
NUM_STEPS = 100000


if __name__ == "__main__":
    for name, model, experiment in MODELS:
        ca = model(experiment)
        start_time = time.time()
        for j in range(NUM_STEPS):
            ca.step()
            time_passed = time.time() - start_time
        speed = NUM_STEPS * ca.cells_num // time_passed
        print("%s: %s cells/s" % (name, sizeof_fmt(speed)))
        del ca
