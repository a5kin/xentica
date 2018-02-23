"""
The module for package-wide RNG.

The main intention is to keep separate deterministic random streams
for every :class:`CellularAutomaton<xentica.core.base.CellularAutomaton>`
instance. So, is you're initialized RNG for a particular CA with some
seed, you're get the guarantee that the random sequence will be the
same, no matter how many other CA's you're running in parallel.

"""
import random
import numpy

__all__ = ['LocalRandom', 'RandInt', ]


class LocalRandom:
    """
    The holder class for the RNG sequence.

    It is incapsulating both standart Python random stream and NumPy one.

    Once instantiated, you can use them as follows::

        from xentica.seeds.random import LocalRandom

        random = LocalRandom()
        # get random number from standard stream
        val = random.std.randint(1, 10)
        # get 100 random numbers from NumPy stream
        vals = random.numpy.randint(1, 10, 100)

    """

    def __init__(self, seed=None):
        """Initialize local random streams."""
        self.std = random.Random(seed)
        np_seed = self.std.getrandbits(32)
        self.np = numpy.random.RandomState(np_seed)

    def load(self, rng):
        """
        Load random state from the class.

        :param rng: :class:`LocalRandom` instance.

        """
        self.std = rng.std
        self.np = rng.np


class RandInt:
    """
    Class, generating a sequence of random integers in some interval.

    It is intended to be used in
    :class:`Experiment <xentica.core.experiments.Experiment>`
    seeds. See the example of initializing CA property above.

    :param min_val: Lower bound for random value.
    :param max_val: Upper bound for random value.

    """

    def __init__(self, min_val, max_val):
        """Initialize the random sequence."""
        self.min_val = min_val
        self.max_val = max_val

    def __get__(self, instance, owner):
        """
        Get the random value in specified range from standard stream.

        This method is used automatically from
        :class:`CellularAutomaton<xentica.core.base.CellularAutomaton>`,
        at the stage of constructing the initial board state.

        """
        return instance.random.std.randint(self.min_val, self.max_val)
