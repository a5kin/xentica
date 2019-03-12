"""Module with bindings to CUDA math functions."""

from xentica.core.variables import DeferredExpression


class Xmath:
    """Static class holding all math functions."""

    @staticmethod
    def min(*args):
        """Calculate the minimum over list of args."""
        raise NotImplementedError

    @staticmethod
    def max(*args):
        """Calculate the maximum over list of args."""
        raise NotImplementedError

    @staticmethod
    def float(val):
        """Cast value to float."""
        return DeferredExpression("((float) (%s))" % val)

    @staticmethod
    def int(val):
        """Cast value to int."""
        return DeferredExpression("((int) (%s))" % val)
