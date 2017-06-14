import random
import numpy


class LocalRandom:

    def __init__(self, seed=None):
        self.std = random.Random(seed)
        np_seed = self.std.getrandbits(32)
        self.np = numpy.random.RandomState(np_seed)


class RandInt:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __get__(self, obj, objtype):
        return random.randint(self.min_val, self.max_val)
