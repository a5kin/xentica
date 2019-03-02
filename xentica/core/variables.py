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

Then you can use them in mixed expressions, like::

    myvar += self.neighbors[i].buffer.state
    self.main.state = myvar & 1

You may also define constants or other ``#define`` patterns with
:class:`Constant` class.

"""
from cached_property import cached_property

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.exceptions import XenticaException

__all__ = ['DeferredExpression', 'Constant', 'Variable', 'IntegerVariable', ]


class DeferredExpression:
    """Base class for other classes intended to be used in mixed expressions.

    In particular, it is used in base
    :class:`Variable <xentica.core.variables.Variable>` and :class:`Property
    <xentica.core.properties.Property>` classes.

    Most of the magic methods dealing with binary and unary operators,
    as well as augmented assigns are automatically overridden for this
    class. As a result, you can use its subclasses in mixed
    expressions with ordinary Python values. See the example in
    module description above.

    Allowed binary ops
        ``+``, ``-``, ``*``, ``/``, ``%``, ``>>``, ``<<``, ``&``,
        ``^``, ``|``, ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``

    Allowed unary ops
        ``+``, ``-``, ``~``, ``abs``, ``int``, ``float``, ``round``

    Allowed augmented assigns
        ``+=``, ``-=``, ``*=``, ``/=``, ``%=``, ``<<=``, ``>>=``,
        ``&=``, ``^=``, ``|=``

    """

    def __init__(self, code=''):
        """Override arithmetic operators and augmented assigns."""
        self.code = code
        binary_ops = (
            ('+', 'add'),
            ('-', 'sub'),
            ('*', 'mul'),
            ('/', 'truediv'),
            ('%', 'mod'),
            ('>>', 'rshift'),
            ('<<', 'lshift'),
            ('&', 'and'),
            ('^', 'xor'),
            ('|', 'or'),
            ('<', 'lt'),
            ('<=', 'le'),
            ('==', 'eq'),
            ('!=', 'ne'),
            ('>', 'gt'),
            ('>=', 'ge'),
        )
        unary_ops = (
            ('-', 'neg'),
            ('+', 'pos'),
            ('abs', 'abs'),
            ('~', 'invert'),
            ('(int)', 'int'),
            ('(float)', 'float'),
            ('round', 'round'),
        )
        for c_op, base_name in binary_ops:
            def binary_direct(oper):
                """Get direct binary operator magic method."""
                def op_func(self_var, value):
                    """Implement direct binary operator."""
                    code = "(%s %s %s)" % (self_var, oper, value)
                    return DeferredExpression(code)
                return op_func

            def binary_reflected(oper):
                """Get reflected binary operator magic method."""
                def op_func(self_var, value):
                    """Implement reflected binary operator."""
                    code = "(%s %s %s)" % (value, oper, self_var)
                    return DeferredExpression(code)
                return op_func

            def augmented_assign(oper):
                """Get augmented assign operator magic method."""
                def op_func(self_var, value):
                    """Implement augmented assign operator."""
                    if not isinstance(self_var, Variable):
                        msg = "Can't assign to DeferredExpression"
                        raise XenticaException(msg)
                    self_var.declare_once()
                    code = "%s %s= %s;\n" % (self_var.var_name, oper, value)
                    self_var.bsca.append_code(code)
                    return self
                return op_func

            func_name = "__%s__" % base_name
            setattr(self.__class__, func_name, binary_direct(c_op))
            func_name = "__r%s__" % base_name
            setattr(self.__class__, func_name, binary_reflected(c_op))
            func_name = "__i%s__" % base_name
            setattr(self.__class__, func_name, augmented_assign(c_op))

        for c_op, base_name in unary_ops:
            def unary(oper):
                """Get unary operator magic method."""
                def op_func(self_var):
                    """Implement unary operator."""
                    code = "(%s(%s))" % (oper, self_var)
                    return DeferredExpression(code)
                return op_func

            func_name = "__%s__" % base_name
            setattr(self.__class__, func_name, unary(c_op))

    def __str__(self):
        """Return the code accumulated in ``self.code``."""
        return self.code


class Constant(BscaDetectorMixin):
    """The class for defining constants and ``#define`` patterns.

    Once you instantiate :class:`Constant`, you must feed it to
    ``BSCA.define_constant()`` in order to generate correct C code::

        const = Constant("C_NAME", "bsca_var")
        self.bsca.define_constant(const)

    :param name:
        Name to use in ``#define``.
    :param value_func:
        Function that take ``BSCA`` instance as argument and return
        the value to be used as second part of ``#define``.

    """

    def __init__(self, name, value_func):
        """Initialize the class."""
        self._name = name
        self._value_func = value_func
        self._pattern_name = name
        self.base_class = Constant

    def get_define_code(self):
        """Get the C code for ``#define``."""
        code = "#define %s {%s}\n" % (self._name, self._pattern_name)
        return code

    def replace_value(self, source):
        """
        Replace the constant's value in generated C code.

        :param source:
            Generated C code.

        """
        val = str(self._value_func(self._holder))
        return source.replace("{%s}" % self._pattern_name, val)

    @property
    def name(self):
        """Get the name of constant."""
        return self._name


class Variable(DeferredExpression, BscaDetectorMixin):
    """
    Base class for all variables.

    Most of the functionality for variables are already implemented in
    it. Though, you are free to re-define it all to implement really
    custom behavior.

    :param val:
         Initial value for the variable.

    """

    def __init__(self, val=None):
        """Initialize base class features."""
        super(Variable, self).__init__()
        self.base_class = Variable
        self._declared = False
        if val is None:
            raise XenticaException("Variable should have initial value.")
        self._init_val = DeferredExpression(str(val))

    @cached_property
    def var_name(self):
        """Get variable name."""
        all_vars = self._holder_frame.f_locals.items()
        for k, var in all_vars:
            if isinstance(var, self.__class__):
                if hash(self) == hash(var):
                    return k
        return "var%d" % abs(hash(self))

    def declare_once(self):
        """Declare variable and assign initial value to it."""
        if not self._declared:
            code = "%s %s = %s;\n" % (
                self.var_type, self.var_name, self._init_val
            )
            self.bsca.append_code(code)
            self._declared = True
            setattr(self.bsca, self.var_name, self)

    def __str__(self):
        """Return a variable name to use in mixed expressions."""
        return self.var_name

    def __get__(self, obj, objtype):
        """Declare a variable on first use."""
        self.declare_once()
        return self

    def __set__(self, obj, value):
        """Assign a new value to variable (doesn't work properly now)."""
        self.declare_once()
        code = "%s = %s;\n" % (self.var_name, value)
        self.bsca.append_code(code)

    @property
    def code(self):
        """Get the variable name as code."""
        return self.var_name

    @code.setter
    def code(self, val):
        """Prevent the change of code."""


class IntegerVariable(Variable):
    """The variable intended to hold a positive integer."""

    #: C type to use in definition.
    var_type = "unsigned int"

    def __init__(self, val=0):
        """Initialize variable with default value."""
        super(IntegerVariable, self).__init__(val)
