import time

from examples.game_of_life import GameOfLife, GOLExperiment
from xentica.utils.formatters import sizeof_fmt


models = [
    ("Conway's Life", GameOfLife, GOLExperiment),
]
num_steps = 10000


if __name__ == "__main__":
    for name, model, experiment in models:
        ca = model(experiment)
        start_time = time.time()
        for j in range(num_steps):
            ca.step()
            time_passed = time.time() - start_time
        speed = num_steps * ca.cells_num // time_passed
        print("%s: %s cells/s" % (name, sizeof_fmt(speed)))
        del ca
