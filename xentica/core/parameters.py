"""
The collection of classes to describe the model's meta-parameters.

The *parameter* is the value that influences the whole model's behavior
in some way. Each parameter has a default value. Then you could
customize them per each experiment or change interactively using
``Bridge``.

There are two types of parameters in Xentica:

Non-interactive
    are constant during a single experiment run. The change of this
    parameter is impossible without a whole model's source code being
    rebuilt. The engine makes sure those params are correctly defined
    globally with the ``#define`` directive. So actually, even if you'll
    change their values at runtime, it doesn't affect the model in any
    way.

Interactive
    could be effectively changed at runtime, since engine traits them
    as extra arguments to CUDA kernels. That means, as long as you'll
    set a new value to an interactive parameter, it will be passed
    into the kernel(s) at the next timestep. Be warned though: every
    parameter declared as interactive, will degrade the model's
    performance further.

The example of parameters' usage::

    from xentica import core
    from xentica.tools.rules import LifeLike
    from examples.game_of_life import GameOfLife, GOLExperiment


    class LifelikeCA(GameOfLife):
        rule = core.Parameter(
            default=LifeLike.golly2int("B3/S23"),
            interactive=True,
        )

        def absorb(self):
            # parent's clone with parameter instead of hardcoded rule
            neighbors_alive = core.IntegerVariable()
            for i in range(len(self.buffers)):
                neighbors_alive += self.neighbors[i].buffer.state
            is_born = (self.rule >> neighbors_alive) & 1
            is_sustain = (self.rule >> 9 >> neighbors_alive) & 1
            self.main.state = is_born | is_sustain & self.main.state


    class DiamoebaExperiment(GOLExperiment):
        rule = LifeLike.golly2int("B35678/S5678")


    model = LifelikeCA(DiamoebaExperiment)

"""
import numpy as np

from xentica.core.variables import Constant
from xentica.core.mixins import BscaDetectorMixin
from xentica.core.expressions import DeferredExpression


class Parameter(BscaDetectorMixin):
    """
    The implementation of Xentica meta-parameter.

    :param default:
        The default value for the parameter to use when it's omitted in
        the experiment class.

    :param interactive:
        ``True`` if the parameter could be safely changed at runtime
        (more details above in the module description).

    """

    def __init__(self, default=0, interactive=False):
        """Initialize the parameter."""
        self._value = default
        self._interactive = interactive
        self._declared = False
        self._name = "param" + str(id(self))
        self._ctypes = {
            int: 'int',
            float: 'float',
            bool: 'bool',
        }
        self._dtypes = {
            int: np.int32,
            float: np.float32,
            bool: bool,
        }

    @property
    def value(self):
        """Get the parameter's value directly."""
        return self._value

    @property
    def name(self):
        """Get the parameter's name."""
        return self._name

    @name.setter
    def name(self, val):
        """Set the parameter's name."""
        self._name = val

    @property
    def ctype(self):
        """Get the parameter's C type."""
        return self._ctypes.get(type(self._value), 'int32')

    @property
    def dtype(self):
        """Get the parameter's NumPy type."""
        return self._dtypes.get(type(self._value), np.int32)

    def _declare_interactive(self):
        """Declare an interactive parameter."""
        if self.bsca.is_parameter(self.name):
            return
        self.bsca.define_parameter(self)

    def _declare_once(self):
        """Declare the parameter when it's mentioned."""
        if self._interactive:
            self._declare_interactive()
            return
        if self._declared:
            return
        self._declared = True
        self.bsca.define_constant(Constant(self.name, self._value))

    def __get__(self, obj, objtype):
        """Implement custom logic when param is get as class descriptor."""
        self._declare_once()
        if self._interactive:
            return DeferredExpression(self._name)
        return self._value

    def __set__(self, obj, value):
        """Implement custom logic when param is set as class descriptor."""
        self._value = value
