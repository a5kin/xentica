import inspect
import itertools

from cached_property import cached_property


class Variable:
    """
    Base class for all variables.

    """
    def __init__(self):
        pass

    @cached_property
    def var_name(self):
        frame = inspect.currentframe().f_back.f_back
        all_vars = itertools.chain(frame.f_globals.items(),
                                   frame.f_locals.items())
        for k, var in all_vars:
            if isinstance(var, self.__class__):
                if hash(self) == hash(var):
                    return k
        return ''


class IntegerVariable(Variable):
    pass


class DeferredExpression:

    def __init__(self, code):
        self.code = code
