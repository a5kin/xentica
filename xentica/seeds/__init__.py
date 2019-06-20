"""
The package for the initial CA state (seed) generation.

Classes from modules below are intended for use in
:class:`Experiment <xentica.core.experiments.Experiment>` classes.

For example, to initialize the whole board with random values::

    from xentica import core, seeds

    class MyExperiment(core.Experiment):
        # ...
        seed = seeds.patterns.PrimordialSoup(
            vals={
                "state": seeds.random.RandInt(0, 1),
            }
       )

"""

from xentica.seeds import random
from xentica.seeds import patterns

__all__ = ['random', 'patterns', ]
