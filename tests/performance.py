"""Script to run performance check."""
import time

from examples.game_of_life import GameOfLife, GOLExperiment
from xentica.utils.formatters import sizeof_fmt


MODELS = [
    ("Conway's Life", GameOfLife, GOLExperiment),
]
NUM_STEPS = 10000


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
