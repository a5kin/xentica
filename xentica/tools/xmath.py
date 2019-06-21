"""The module with bindings to CUDA math functions."""

from xentica.core.expressions import DeferredExpression


class Xmath:
    """Static class holding all math functions."""

    @staticmethod
    def min(*args):
        """Calculate the minimum over list of args."""
        expr = str(args[0])
        for arg in args[1:]:
            expr = "(({a}) < ({b})) ? ({a}) : ({b})".format(a=expr, b=arg)
        return DeferredExpression("(%s)" % expr)

    @staticmethod
    def max(*args):
        """Calculate the maximum over list of args."""
        expr = str(args[0])
        for arg in args[1:]:
            expr = "(({a}) > ({b})) ? ({a}) : ({b})".format(a=expr, b=arg)
        return DeferredExpression("(%s)" % expr)

    @staticmethod
    def float(val):
        """Cast a value to float."""
        return DeferredExpression("((float) (%s))" % val)

    @staticmethod
    def int(val):
        """Cast a value to int."""
        return DeferredExpression("((unsigned int) (%s))" % val)

    @staticmethod
    def popc(val):
        """Count the number of bits that are set to '1' in a 32 bit integer."""
        return DeferredExpression("(__popc(%s))" % val)
