"""
The module for package-wide RNG.

The main intention is to keep separate deterministic random streams
for every :class:`CellularAutomaton<xentica.core.base.CellularAutomaton>`
instance. So, if you've initialized RNG for a particular CA with some
seed, you're geting the guarantee that the random sequence will be the
same, no matter how many other CA's you're running in parallel.

"""
import random
import functools
import operator

import numpy as np

from xentica.core.expressions import PatternExpression

__all__ = ['LocalRandom', 'RandInt', ]


class LocalRandom:
    """
    The holder class for the RNG sequence.

    It is encapsulating both standard Python random stream and NumPy one.

    Once instantiated, you can use them as follows::

        from xentica.seeds.random import LocalRandom

        random = LocalRandom()
        # get random number from standard stream
        val = random.standard.randint(1, 10)
        # get 100 random numbers from NumPy stream
        vals = random.numpy.randint(1, 10, 100)

    """

    def __init__(self, seed=None):
        """Initialize local random streams."""
        self.standard = random.Random(seed)
        np_seed = self.standard.getrandbits(32)
        self.numpy = np.random.RandomState(np_seed)

    def load(self, rng):
        """
        Load a random state from the class.

        :param rng: :class:`LocalRandom` instance.

        """
        self.standard = rng.standard
        self.numpy = rng.numpy


class RandInt(PatternExpression):
    """
    Class, generating a sequence of random integers in some interval.

    It is intended for use in
    :class:`Experiment <xentica.core.experiments.Experiment>`
    seeds. See the example of initializing CA property above.

    :param min_val:
         Lower bound for a random value.
    :param max_val:
         Upper bound for a random value.
    :param constant:
         If ``True``, will force the use of the standard random stream.

    """

    def __init__(self, min_val, max_val, constant=False):
        """Initialize the random sequence."""
        super(RandInt, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self._constant = constant

    def __get__(self, instance, owner):
        """
        Get random value(s) in specified range from Numpy or standard stream.

        This method is used automatically from
        :class:`CellularAutomaton<xentica.core.base.CellularAutomaton>`,
        at the stage of constructing the initial board state.

        """
        if hasattr(instance, "size") and not self._constant:
            num_values = functools.reduce(operator.mul, instance.size)
            return instance.random.numpy.randint(self.min_val,
                                                 self.max_val + 1,
                                                 num_values)
        instance = instance or owner
        return instance.random.standard.randint(self.min_val, self.max_val)
