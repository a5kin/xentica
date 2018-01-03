"""
The collection of classes to describe experiments for CA models.

Experiment is a class with CA parameters stored as class
variables. Different models may have a different set of parameters. To
make sure all set correct, you should inherit your experiments from
:class:`Experiment` class.

The quick example::

    from xentica import core, seeds

    class MyExperiment(core.Experiment):
        # RNG seed string
        word = "My Special String"
        # field size
        size = (640, 360, )
        # initial field zoom
        zoom = 3
        # initial field shift
        pos = [0, 0]
        # A pattern used in initial board state generation.
        # BigBang is a small area initialized with high-density random values.
        seed = seeds.patterns.BigBang(
            # position Big Bang area
            pos=(320, 180),
            # size of Big Bang area
            size=(100, 100),
            # algorithm to generate random values
            vals={
                "state": seeds.random.RandInt(0, 1),
            }
       )



"""


class Experiment:
    """
    Base class for all experiments.

    Right now doing nothing, but will be improved in future
    versions. So it is adviced to inherit your experiments from it.

    """
