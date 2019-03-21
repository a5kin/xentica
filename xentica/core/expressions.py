"""Module holding base ``DeferredExpression`` class."""
from xentica.core.exceptions import XenticaException

__all__ = ['DeferredExpression', ]


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
                    if type(self_var).__name__ == "DeferredExpression":
                        msg = "Can't assign to DeferredExpression"
                        raise XenticaException(msg)
                    self_var.declare_once()
                    code = "%s %s= %s;\n" % (self_var.var_name, oper, value)
                    self_var.bsca.append_code(code)
                    has_fallback = hasattr(self_var, "fallback_name")
                    is_default = self_var.fallback_name == "var"
                    if has_fallback and is_default:
                        self_var.fallback_name = self.var_name
                    return self_var
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
