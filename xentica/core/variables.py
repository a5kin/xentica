"""
The collection of classes to declare and use C variables and constants.

If the logic of your ``emit()``, ``absorb()`` or ``color()`` functions
requires the intermediate variables, you must declare them via classes
from this module in the following way::

    from xentica import core

    class MyCA(core.CellularAutomaton):
        # ...

        def emit(self):
            myvar = core.IntegerVariable()

Then you can use them in mixed expressions, like this::

    myvar += self.neighbors[i].buffer.state
    self.main.state = myvar & 1

You may also define constants or other ``#define`` patterns with
:class:`Constant` class.

"""
from cached_property import cached_property

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.exceptions import XenticaException
from xentica.core.expressions import DeferredExpression
from xentica.core.base import CellularAutomaton

__all__ = [
    'Constant', 'Variable',
    'IntegerVariable', 'FloatVariable',
]


class Constant(BscaDetectorMixin):
    """
    The class for defining constants and ``#define`` patterns.

    Once you instantiate :class:`Constant`, you must feed it to
    ``CellularAutomaton.define_constant()`` in order to generate
    the correct C code::

        const = Constant("C_NAME", "some_value")
        self.bsca.define_constant(const)

    :param name:
        The name to use in ``#define``.
    :param value:
        A value for the define, it will be converted to a string
        with ``str()``.

    """

    def __init__(self, name, value):
        """Initialize the class."""
        self._name = name
        self._value = value
        self.base_class = Constant

    def get_define_code(self):
        """Get the C code for ``#define``."""
        code = "#define %s %s\n" % (self._name, str(self._value))
        return code

    @property
    def name(self):
        """Get the name of the constant."""
        return self._name


class Variable(DeferredExpression, BscaDetectorMixin):
    """
    The base class for all variables.

    Most of the functionality for variables is already implemented in
    it. Though, you are free to re-define it all to implement really
    custom behavior.

    :param val:
         The initial value for the variable.
    :param name:
         Fallback name to declare the variable with.

    """

    def __init__(self, val=None, name="var"):
        """Initialize base class features."""
        super().__init__()
        self.fallback_name = name
        self.base_class = Variable
        if val is None:
            raise XenticaException("Variable should have an initial value.")
        self._init_val = DeferredExpression(str(val))

    @cached_property
    def var_name(self):
        """Get the variable name."""
        all_vars = self._holder_frame.f_locals.items()
        model = None
        bad_names = ("self_var", "obj", "cls")
        for k, var in all_vars:
            if isinstance(var, CellularAutomaton):
                model = var
            if isinstance(var, self.__class__):
                if hash(self) == hash(var) and k not in bad_names:
                    return k
        if model is not None and self.fallback_name == "var":
            for k, var in model.__class__.__dict__.items():
                if isinstance(var, self.__class__):
                    if hash(self) == hash(var) and k not in bad_names:
                        return k
        return self.fallback_name

    def declare_once(self):
        """Declare the variable and assign the initial value to it."""
        if not self.bsca.is_declared(self):
            code = "%s %s = %s;\n" % (
                self.var_type, self.var_name, self._init_val
            )
            self.bsca.append_code(code)
            self.bsca.declare(self)
            setattr(self.bsca, self.var_name, self)

    def __str__(self):
        """Return the variable name to use in mixed expressions."""
        return self.var_name

    def __get__(self, obj, objtype):
        """Declare the variable on first use."""
        self.declare_once()
        return self

    def __set__(self, obj, value):
        """Assign a new value to variable (doesn't work properly now)."""
        self.declare_once()
        if str(self.var_name) == str(value):
            return
        code = "%s = %s;\n" % (self.var_name, value)
        self.bsca.append_code(code)

    @property
    def code(self):
        """Get the variable name as a C code."""
        return self.var_name

    @code.setter
    def code(self, val):
        """Prevent the change of the code."""


class IntegerVariable(Variable):
    """The variable intended to hold a positive integer."""

    #: C type to use in definition.
    var_type = "unsigned int"

    def __init__(self, val="0", **kwargs):
        """Initialize a variable with the default value."""
        super().__init__(val, **kwargs)


class FloatVariable(Variable):
    """The variable intended to hold a 32-bit float."""

    #: C type to use in definition.
    var_type = "float"

    def __init__(self, val="0.0f", **kwargs):
        """Initialize a variable with the default value."""
        super().__init__(val, **kwargs)
